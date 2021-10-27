import io
import numpy as np
from PIL import Image

# pytorch
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.retinanet import RetinaNet

# xaitk-saliency
from smqtk_classifier import ClassifyImage
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

# postprocess detections
from .postprocess import faster_rcnn_postprocess_detections, retinanet_postprocess_detections

# TODO:
# 1) Figure out top-k behavior for different models
# 2) Fix string/number saliency input params
# 3) Toggle for boolean 'debiased' parameter
# 4) Add multiple components for non-classification approaches (PerturbImage and Scoring methods)
# 5) Work out RetinaNet monkey patch


AI_MAP = {
    "resnet-50": models.resnet50(pretrained=True),
    "alexnet": models.alexnet(pretrained=True),
    "vgg-16": models.vgg16(pretrained=True),
    "faster-rcnn": models.detection.fasterrcnn_resnet50_fpn(pretrained=True),
    "retina-net": models.detection.retinanet_resnet50_fpn(pretrained=True),
}

XAI_MAP = {
    "RISEStack": RISEStack,
    "SlidingWindowStack": SlidingWindowStack,
}

# Pytorch pre-processing
model_loader = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)


# SMQTK black-box classifier
class ClfModel(ClassifyImage):
    def __init__(self, model, idx):
        self.model = model
        self.idx = idx

    def get_labels(self):
        return ["output"]

    def classify_images(self, image_iter):
        for img in image_iter:
            inp = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
            out = self.model(inp)[0, self.idx].detach().numpy()
            yield dict(zip(self.get_labels(), [out]))

    def get_config(self):
        return {}


class XaiController:
    def __init__(self):
        self._task = None
        self._model = None
        self._saliency = None
        self._image_1 = None
        self._image_2 = None
        self._saliency_params = {}

    def set_task(self, task_name):
        self._task = task_name

    def set_model(self, model_name):
        try:
            self._model = AI_MAP[model_name]
            self._model.eval()
            # Perform monkey patch to get class probabilities
            if self._task == 'detection':
                if model_name == 'faster-rcnn':
                    RoIHeads.postprocess_detections = faster_rcnn_postprocess_detections
                else:
                    RetinaNet.postprocess_detections = retinanet_postprocess_detections
            # Perform model surgery to get feature descriptors
            elif self._task == 'similarity':
                if model_name == 'resnet-50':
                    self._model = nn.Sequential(
                        *list(self._model.children())[:-1])
                else:
                    self._model.classifier = nn.Sequential(
                        *list(self._model.classifier.children())[:-1])
        except:
            print(f"Could not find {model_name} in {AI_MAP.keys()}")

    def set_saliency_method(self, method_name, params):
        try:
            self._saliency = XAI_MAP[method_name](**params)
        except:
            print(f"Could not find {method_name} in {XAI_MAP.keys()}")

    def set_image_1(self, bytes_content):
        self._image_1 = np.array(Image.open(io.BytesIO(bytes_content)))

    def set_image_2(self, bytes_content):
        self._image_2 = np.array(Image.open(io.BytesIO(bytes_content)))

    def run_model(self):
        # Get model predictions
        preds = self._model(model_loader(self._image_1).unsqueeze(0))

        if self._task == 'classification':
            pass
        elif self._task == 'detection':
            pass
        elif self._task == 'similarity':
            pass

    def run_saliency(self):
        if self._task == 'classification':
            sal = self._saliency(self._image_1, ClfModel(self._model))
        elif self._task == 'detection':
            # perturb, then saliency generation
            sal = self._saliency(self._image_1, self._model)
        elif self._task == 'similarity':
            # perturb, then saliency generation
            sal = self._saliency(
                self._image_1, self._image_2, self._model)

    def update_saliency_params(self, **kwargs):
        self._saliency_params.update(kwargs)
