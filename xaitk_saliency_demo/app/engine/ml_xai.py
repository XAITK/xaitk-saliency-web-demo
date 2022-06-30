from typing import Dict, Any, Iterable
import numpy as np
from scipy.special import softmax

# pytorch
import torch

# xaitk-saliency
from xaitk_saliency.impls.perturb_image.rise import RISEGrid
from xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring import DRISEScoring

from xaitk_saliency.impls.gen_image_similarity_blackbox_sal.sbsm import SBSMStack
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import RISEStack
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import (
    SlidingWindowStack,
)
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import RandomGridStack

# smqtk-*
from smqtk_classifier import ClassifyImage
from smqtk_descriptors.interfaces.image_descriptor_generator import ImageDescriptorGenerator
from smqtk_detection import DetectImageObjects

# labels
from .assets import imagenet_categories, imagenet_model_loader

import logging

logger = logging.getLogger("xaitks_saliency_demo")

# Model source dataset's mean pixel value. It just so happens that the models
# listed in `ml_models.py`. If this becomes violated then there will need to be
# a mechanism to get this on a per-model basis.
FILL = np.uint8(np.asarray([0.485, 0.456, 0.406]) * 255)

# Mapping of labels to variable construction metadata used in `Saliency` class
# instantiation. This ultimately maps a saliency task label to the algorithm
# instance to be used.
SALIENCY_TYPES = {
    # Classification
    "RISEStack": {
        "_saliency": {
            "class": RISEStack,
        },
    },
    "SlidingWindowStack": {
        "_saliency": {
            "class": SlidingWindowStack,
        },
    },

    # Similarity
    "SBSMStack": {
        "_saliency": {
            "class": SBSMStack,
        }
    },

    # Object Detection
    "DRISEStack": {
        "_saliency": {
            "class": DRISEStack
        }
    },
    "RandomGridStack": {
        "_saliency": {
            "class": RandomGridStack
        }
    },
}


# SMQTK black-box classifier
class ClfModel(ClassifyImage):
    def __init__(self, model, idx):
        self.model = model
        self.idx = idx

    def get_labels(self):
        return [imagenet_categories[i] for i in self.idx]

    @torch.no_grad()
    def classify_images(self, image_iter):
        for img in image_iter:
            inp = imagenet_model_loader(img).unsqueeze(0).to(self.model.device)
            vec = self.model(inp).cpu().numpy().squeeze()
            out = softmax(vec)
            yield dict(zip(self.get_labels(), out[self.idx]))

    def get_config(self):
        # Required by a parent class. Will not be used in this context.
        return {}


# SMQTK black-box descriptor generator
class DescrModel(ImageDescriptorGenerator):

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def generate_arrays_from_images(self, img_mat_iter: Iterable[np.ndarray]) -> Iterable[np.ndarray]:
        for img in img_mat_iter:
            inp = imagenet_model_loader(img).unsqueeze(0).to(self.model.device)
            vec = self.model(inp).cpu().numpy().squeeze()
            yield vec

    def get_config(self) -> Dict[str, Any]:
        # Required by a parent class. Will not be used in this context.
        return {}


# -----------------------------------------------------------------------------
class Saliency:
    def __init__(self, model, name, params):
        self._model = model
        try:
            kv_pairs = SALIENCY_TYPES[name].items()
        except IndexError:
            logger.info(f"Could not find {name} in {list(SALIENCY_TYPES.keys())}")
            return
        for key, value in kv_pairs:
            constructor = value.get("class")
            param_keys = value.get("params", params.keys())
            setattr(self, key, constructor(**{k: params[k] for k in param_keys}))


class ClassificationSaliency(Saliency):
    def run(self, input, *_):
        topk = self._model.topk
        self._saliency.fill = FILL
        sal = self._saliency(input, ClfModel(self._model, topk))
        return {
            "type": "classification",
            "saliency": sal,
        }


class SimilaritySaliency(Saliency):
    def run(self, reference, query):
        self._saliency.fill = FILL
        sal = self._saliency(reference, [query], DescrModel(self._model))
        return {
            "type": "similarity",
            "saliency": sal,
        }


class DetectionSaliency(Saliency):
    def run(self, input, *_):
        # Generate reference image detections
        detector: DetectImageObjects = self._model.model
        bboxes, scores, _ = self._model.predict(input)
        # Trim to top-k
        topk = self._model.topk
        bboxes = bboxes[topk]
        scores = scores[topk]
        # Generate saliency maps
        self._saliency.fill = FILL
        sal = self._saliency(input, bboxes, scores, detector)

        return {
            "type": "detection",
            "saliency": sal,
        }


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

SALIENCY_BASE_CLASSES = {
    "similarity": SimilaritySaliency,
    "detection": DetectionSaliency,
    "classification": ClassificationSaliency,
}


def get_saliency(task_name, model, name, params):
    constructor = SALIENCY_BASE_CLASSES[task_name]
    return constructor(model, name, params)
