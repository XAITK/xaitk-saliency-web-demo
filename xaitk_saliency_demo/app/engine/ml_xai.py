import numpy as np
from scipy.special import softmax

# pytorch
import torch
import torch.nn as nn

# xaitk-saliency
from xaitk_saliency.impls.perturb_image.rise import RISEGrid
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow
from xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring import DRISEScoring
from xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring import (
    SimilarityScoring,
)
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import RISEStack
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import (
    SlidingWindowStack,
)

# xaitk-saliency
from smqtk_classifier import ClassifyImage
from xaitk_saliency.utils.masking import occlude_image_batch

# labels
from .assets import imagenet_categories, imagenet_model_loader

FILL = np.uint8(np.asarray([0.485, 0.456, 0.406]) * 255)

SALIENCY_TYPES = {
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
    "similarity-saliency": {
        "_perturb": {
            "class": SlidingWindow,
            "params": ["window_size", "stride"],
        },
        "_saliency": {
            "class": SimilarityScoring,
            "params": ["proximity_metric"],
        },
    },
    "detection-saliency": {
        "_perturb": {
            "class": RISEGrid,
            "params": ["n", "s", "p1", "seed", "threads"],
        },
        "_saliency": {
            "class": DRISEScoring,
            "params": ["proximity_metric"],
        },
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
        # Required by a parent class.
        return {}


class Saliency:
    def __init__(self, model, name, params):
        self._model = model
        try:
            for key, value in SALIENCY_TYPES[name].items():
                constructor = value.get("class")
                param_keys = value.get("params", params.keys())
                setattr(self, key, constructor(**{k: params[k] for k in param_keys}))
        except:
            print(f"Could not find {name} in {list(SALIENCY_TYPES.keys())}")


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
    def run(self, query, reference):
        # generate query/reference features
        query_feat = self._model.predict(query)
        ref_feat = self._model.predict(reference)
        # generate perturbed features
        pert_masks = self._perturb(reference)
        pert_ref_imgs = occlude_image_batch(reference, pert_masks, FILL)
        pert_ref_feats = np.asarray([self._model.predict(pi) for pi in pert_ref_imgs])
        # generate saliency map
        sal = self._saliency(query_feat, ref_feat, pert_ref_feats, pert_masks)
        return {
            "type": "similarity",
            "references": ref_feat,
            "masks": pert_masks,
            "predictions": pert_ref_feats,
            "saliency": sal,
        }


class DetectionSaliency(Saliency):
    def run(self, input, *_):
        # generate reference prediction
        topk = self._model.topk
        ref_preds = self._model.predict(input)[topk, :]
        # generate perturbed predictions
        pert_masks = self._perturb(input)
        pert_imgs = occlude_image_batch(input, pert_masks, FILL)
        pert_preds = np.asarray([self._model.predict(pi) for pi in pert_imgs])
        # generate saliency map
        sal = self._saliency(ref_preds, pert_preds, pert_masks)

        return {
            "type": "detection",
            "references": ref_preds,
            "masks": pert_masks,
            "predictions": pert_preds,
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
