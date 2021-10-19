import io
import numpy as np
from PIL import Image

# pytorch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

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

AI_MAP = {
    "resnet-50": models.resnet50(pretrained=True),
    "alexnet": models.alexnet(pretrained=True),
    "vgg-16": models.vgg16(pretrained=True),
    "faster-rcnn": models.detection.fasterrcnn_resnet50_fpn(pretrained=True),
    "retina-net": models.detection.retinanet_resnet50_fpn(pretrained=True),
}

# Pytorch pre-processing
model_loader = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class XaiController:
    def __init__(self):
        self._model = None
        self._image_1 = None
        self._image_2 = None
        self._saliency_params = {}

    def set_model(self, model_name):
        try:
            self._model = AI_MAP[model_name]
        except:
            print(f"Could not find {model_name} in {AI_MAP.keys()}")

    def set_saliency_method(self, method_name):
        pass

    def set_image_1(self, bytes_content):
        self._image_1 = Image.open(io.BytesIO(bytes_content))

    def set_image_2(self, bytes_content):
        self._image_2 = Image.open(io.BytesIO(bytes_content))

    def run_model(self):
        self._model(model_loader(self._image_1))

    def update_saliency_params(self, **kwargs):
        self._saliency_params.update(kwargs)
