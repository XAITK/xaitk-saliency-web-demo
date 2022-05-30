import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

# pytorch
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.retinanet import RetinaNet

# xaitk-saliency
from xaitk_saliency.utils.detection import format_detection

# postprocess detections
from .postprocess import (
    faster_rcnn_postprocess_detections,
    retinanet_postprocess_detections,
)

# labels + transform
from .assets import (
    imagenet_categories,
    coco_categories,
    coco_valid_idxs,
    imagenet_model_loader,
    coco_model_loader,
)

import logging

logger = logging.getLogger("xaitks_saliency_demo")

# -----------------------------------------------------------------------------

DEVICE = torch.device("cpu")


def update_ml_device(cpu_only=True):
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
    "DetectionFasterRCNN",
    "DetectionRetinaNet",
]

# -----------------------------------------------------------------------------
# Perform monkey patch to get class probabilities
# -----------------------------------------------------------------------------

RoIHeads.postprocess_detections = faster_rcnn_postprocess_detections
RetinaNet.postprocess_detections = retinanet_postprocess_detections

# -----------------------------------------------------------------------------
# Class normalizing model usage
# -----------------------------------------------------------------------------


class AbstractModel:
    def __init__(self, server, model, device=DEVICE):
        self._server = server
        self._device = device
        self.topk = 10
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


class DetectionPredict:
    @torch.no_grad()
    def predict(self, input):
        input = coco_model_loader(input).unsqueeze(0)
        input = input.to(self.device)
        output = self._model(input)[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"][:, coco_valid_idxs].cpu().numpy()
        return format_detection(boxes, scores)  # .astype('float32')


class ResNetPredict:
    @torch.no_grad()
    def predict(self, input):
        input = imagenet_model_loader(input).unsqueeze(0)
        input = input.to(self.device)
        output = self._model(input).squeeze()
        return output.cpu().numpy()


# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------


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


class DetectionRun:
    def run(self, input, *_):
        preds = self.predict(input)
        boxes = preds[:, :4]
        scores = np.max(preds[:, 5:], axis=1)
        scores_idx = np.argmax(preds[:, 5:], axis=1)
        topk = np.argsort(-scores)[: self.state.TOP_K]
        output = [(coco_categories[scores_idx[i]], scores[i], boxes[i]) for i in topk]

        # store for later
        self.topk = topk

        logger.info(f"Predicted bounding boxes: {output}")
        return {
            "type": "detection",
            "topk": topk,
            "detection": output,
        }


class DetectionFasterRCNN(AbstractModel, DetectionPredict, DetectionRun):
    def __init__(self, server):
        super().__init__(
            server,
            models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True,
                box_score_thresh=0.0,
            ),
        )


class DetectionRetinaNet(AbstractModel, DetectionPredict, DetectionRun):
    def __init__(self, server):
        super().__init__(
            server,
            models.detection.retinanet_resnet50_fpn(
                pretrained=True,
                score_thresh=0.0,
                detections_per_img=100,
            ),
        )


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
    except:
        logger.info(f"Could not find {model_name} in {globals().keys()}")

    return MODEL_INSTANCES[server].get(model_name)
