from typing import List
from typing import Tuple

import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from ubelt import grabdata

# pytorch
import torch
import torch.nn as nn
import torchvision.models as models

# xaitk-saliency
from smqtk_detection.impls.detect_image_objects.resnet_frcnn import ResNetFRCNN
from smqtk_detection.impls.detect_image_objects.centernet import CenterNetVisdrone

# labels + transform
from .assets import (
    imagenet_categories,
    imagenet_model_loader,
)

import logging

logger = logging.getLogger("xaitks_saliency_demo")

# -----------------------------------------------------------------------------

DEVICE = torch.device("cpu")


def update_ml_device(cpu_only=True):
    global DEVICE
    if not cpu_only and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        logger.info(" ~~~ Using GPU ~~~ \n")
    else:
        logger.info(" ~~~ Using CPU ~~~ \n")


# -----------------------------------------------------------------------------

__all__ = [
    # API ------------------
    "get_model",
    # classification -------
    "ClassificationResNet50",
    "ClassificationAlexNet",
    "ClassificationVgg16",
    # similarity -----------
    "SimilarityResNet50",
    "SimilarityAlexNet",
    "SimilarityVgg16",
    # detection ------------
    "DetectionFRCNN",
    "DetectionCenterNetVisdrone",
]

# -----------------------------------------------------------------------------
# Model file acquisition / caching
# -----------------------------------------------------------------------------

CENTERNET_RESNET50 = grabdata(
    "https://data.kitware.com/api/v1/item/623259f64acac99f426f21db/download",
    fname="centernet-resnet50.pth",
    appname="xaitk-saliency-demo",
    hash_prefix="a0083ec55d46c420d06c414e5ecc1863d6ad9b6a1732acff5c9dba28158a4"
                "c5a04f43541415d503fa776031a7329e3912864ae2348b3bee035df0d1e7a"
                "cefa49"
)

# -----------------------------------------------------------------------------
# Class normalizing model usage
# -----------------------------------------------------------------------------


class AbstractModel:
    def __init__(self, server, model, device=None):
        if device is None:
            device = DEVICE
        self._server = server
        self._device = device
        self.topk = 10
        self._model = model
        if isinstance(model, torch.nn.Module):
            self._model = model.to(self._device)
            self._model.eval()

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @property
    def device(self):
        return self._device

    @property
    def state(self):
        return self._server.state

    @property
    def model(self):
        return self._model


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------

class ResNetPredict:
    @torch.no_grad()
    def predict(self, input) -> np.ndarray:
        """
        Run prediction with the set model and return the final output layer
        as a vector for one image.
        """
        input = imagenet_model_loader(input).unsqueeze(0)
        input = input.to(self.device)
        output = self._model(input).squeeze()
        return output.cpu().numpy()


class ClassificationRun:
    def run(self, input, *_):
        preds = self.predict(input)
        preds = softmax(preds)
        topk = np.argsort(-preds)[: self.state.TOP_K]
        output = [(imagenet_categories[i], preds[i]) for i in topk]

        # store for later
        self.topk = topk

        logger.info(f"Predicted classes: {output}")
        return {
            "type": "classification",
            "topk": topk,
            "classes": output,
        }


class ClassificationResNet50(AbstractModel, ResNetPredict, ClassificationRun):
    def __init__(self, server):
        super().__init__(server, models.resnet50(pretrained=True))


class ClassificationAlexNet(AbstractModel, ResNetPredict, ClassificationRun):
    def __init__(self, server):
        super().__init__(server, models.alexnet(pretrained=True))


class ClassificationVgg16(AbstractModel, ResNetPredict, ClassificationRun):
    def __init__(self, server):
        super().__init__(server, models.vgg16(pretrained=True))


# -----------------------------------------------------------------------------
# Similarity
# -----------------------------------------------------------------------------


class SimilarityRun:
    def run(self, query, reference, *_):
        p_query = self.predict(query)
        p_reference = self.predict(reference)
        similarity = cosine_similarity(
            p_query.reshape(1, -1),
            p_reference.reshape(1, -1),
        ).item()
        logger.info(f"Similarity score: {similarity}")
        return {
            "type": "similarity",
            "similarity": similarity,
        }


class SimilarityResNet50(AbstractModel, ResNetPredict, SimilarityRun):
    def __init__(self, server):
        super().__init__(server, models.resnet50(pretrained=True))
        # Perform model surgery to get feature descriptors
        self._model = nn.Sequential(*list(self._model.children())[:-1])


class SimilarityAlexNet(AbstractModel, ResNetPredict, SimilarityRun):
    def __init__(self, server):
        super().__init__(server, models.alexnet(pretrained=True))
        # Perform model surgery to get feature descriptors
        self._model.classifier = nn.Sequential(
            *list(self._model.classifier.children())[:-1]
        )


class SimilarityVgg16(AbstractModel, ResNetPredict, SimilarityRun):
    def __init__(self, server):
        super().__init__(server, models.vgg16(pretrained=True))
        # Perform model surgery to get feature descriptors
        self._model.classifier = nn.Sequential(
            *list(self._model.classifier.children())[:-1]
        )


# -----------------------------------------------------------------------------
# Detection
# -----------------------------------------------------------------------------

class DetectionPredict:
    def predict(self, input: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Predict detections for a single image, returning bounding-boxes, scores
        and labels as arrays (in that order).
        Bounding boxes are in (minX, minY, maxX, maxY) format.
        Boxes array is of shape (nDets x 4).
        Scores array is of shape (nDets x nClasses).
        Labels list is of length (nClasses).
        """
        preds = list(list(self._model([input]))[0])
        n_preds = len(preds)
        # TODO: What should happen when there are no detections in the image?
        n_classes = len(preds[0][1])
        # scores matrix should be ordered the same as `list(dict.keys())`
        bboxes = np.empty((n_preds, 4), dtype=np.float32)
        scores = np.empty((n_preds, n_classes), dtype=np.float32)
        labels = None
        for i, (bbox, score_dict) in enumerate(preds):
            bboxes[i] = (*bbox.min_vertex, *bbox.max_vertex)
            scores[i] = list(score_dict.values())
            if labels is None:
                labels = list(score_dict.keys())

        return bboxes, scores, labels


class DetectionRun:
    def run(self, input: np.ndarray, *_):
        """
        Generic "run" function for producing top-K image object detections.
        """
        boxes, scores, labels = self.predict(input)
        max_scores = np.max(scores, axis=1)
        max_scores_idx = np.argmax(scores, axis=1)
        topk = np.argsort(-max_scores)[: self.state.TOP_K]
        output = [(labels[max_scores_idx[i]], max_scores[i], boxes[i]) for i in topk]
        # output is of format
        # [ ..., (category-label, score, LTRB bbox array), ... ]

        # store for later
        self.topk = topk

        logger.info(f"Predicted bounding boxes: {output}")
        return {
            "type": "detection",
            "topk": topk,
            "detection": output,
        }


class DetectionFRCNN(AbstractModel, DetectionPredict, DetectionRun):
    def __init__(self, server):
        d = DEVICE
        model = ResNetFRCNN(
            use_cuda=True if 'cuda' in d.type.lower() else False,
            cuda_device=d.type + (f":{d.index}" if d.index is not None else ""),
        )
        super().__init__(server, model)


class DetectionCenterNetVisdrone(AbstractModel, DetectionPredict, DetectionRun):
    def __init__(self, server):
        d = DEVICE
        model = CenterNetVisdrone(
            arch="resnet50",
            model_file=CENTERNET_RESNET50,
            max_dets=500,
            use_cuda=True if 'cuda' in d.type.lower() else False,
        )
        super().__init__(server, model)


# -----------------------------------------------------------------------------
# Factory instance maps
# -----------------------------------------------------------------------------

MODEL_INSTANCES = {}

# -----------------------------------------------------------------------------
# Factory methods
# -----------------------------------------------------------------------------


def get_model(server, model_name):
    global MODEL_INSTANCES
    MODEL_INSTANCES.setdefault(server, {})

    if model_name in MODEL_INSTANCES[server]:
        return MODEL_INSTANCES[server][model_name]

    try:
        constructor = globals()[model_name]
        MODEL_INSTANCES[server][model_name] = constructor(server)
    except KeyError:
        logger.info(f"Could not find {model_name} in {globals().keys()}")

    return MODEL_INSTANCES[server].get(model_name)
