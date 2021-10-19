from trame import change, get_state, update_state

import io
import base64
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
from xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring import SimilarityScoring
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import RISEStack
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import SlidingWindowStack


# -----------------------------------------------------------------------------
# Application logic
# Task > Model > Saliency
# -----------------------------------------------------------------------------

TASKS = [
    {"text": "Image Similarity", "value": "similarity"},
    {"text": "Object Detection", "value": "detection"},
    {"text": "Image Classification", "value": "classification"},
]

TASK_DEPENDENCY = {
    "similarity": {
        # Task => saliency
        "saliency_active": "SlidingWindow",
        "saliency_available": [
            {"text": "Sliding Window", "value": "SlidingWindow"},
            {"text": "Similarity Scoring", "value": "SimilarityScoring"},
        ],
        # Task => model
        "model_active": "resnet-50",
        "model_available": [
            {"text": "ResNet-50", "value": "resnet-50"},
            {"text": "AlexNet", "value": "alexnet"},
            {"text": "VGG-16", "value": "vgg-16"},
        ],
        # Task => input
        "image_count": 2,
    },
    "detection": {
        # Task => saliency
        "saliency_active": "RISEGrid",
        "saliency_available": [
            {"text": "RISE Grid", "value": "RISEGrid"},
            {"text": "DRISE Scoring", "value": "DRISEScoring"},
        ],
        # Task => model
        "model_active": "faster-rcnn",
        "model_available": [
            {"text": "Faster R-CNN", "value": "faster-rcnn"},
            {"text": "RetinaNet", "value": "retina-net"},
        ],
        # Task => input
        "image_count": 1,
    },
    "classification": {
        # Task => saliency
        "saliency_active": "RISEStack",
        "saliency_available": [
            {"text": "RISE Stack", "value": "RISEStack"},
            {"text": "Sliding Window Stack", "value": "SlidingWindowStack"},
        ],
        # Task => model
        "model_active": "resnet-50",
        "model_available": [
            {"text": "ResNet-50", "value": "resnet-50"},
            {"text": "AlexNet", "value": "alexnet"},
            {"text": "VGG-16", "value": "vgg-16"},
        ],
        # Task => input
        "image_count": 1,
    },
}

SALIENCY_PARAMS = {
    "SlidingWindow": ["window_size", "stride"],
    "SimilarityScoring": ["similarity_metric"],
    "RISEGrid": ["n", "s", "p1", "seed", "threads"],
    "DRISEScoring": ["proximity_metric"],
    "RISEStack": ["n", "s", "p1", "seed", "threads", "debiased"],
    "SlidingWindowStack": ["window_size", "stride", "threads"],
}

AI_MAP = {
    "resnet-50": models.resnet50(pretrained=True),
    "alexnet": models.alexnet(pretrained=True),
    "vgg-16": models.vgg16(pretrained=True),
    "faster-rcnn": models.detection.fasterrcnn_resnet50_fpn(pretrained=True),
    "retina-net": models.detection.retinanet_resnet50_fpn(pretrained=True),
}


@change("task_active")
def task_change(task_active, **kwargs):
    """Task is changing"""
    if task_active in TASK_DEPENDENCY:
        for key, value in TASK_DEPENDENCY[task_active].items():
            update_state(key, value)

    # Reset client state
    update_state("need_input", True)
    update_state("image_url_1", None)
    update_state("image_url_2", None)
    update_state("predict_url", None)


@change("model_active")
def model_change(model_active, **kwargs):
    """ML model is changing"""
    if model_active in AI_MAP:
        print(f"Use model {model_active}")
        model = AI_MAP[model_active]

        # if task_active == 'similarity':
        #     model = nn.Sequential(**model.children()[-1])

        return model

    print(f"Could not find {model_active} in {AI_MAP.keys()}")


@change("saliency_active")
def saliency_change(saliency_active, **kwargs):
    """Saliency algo is changing"""
    print("Use saliency", saliency_active)
    update_state("saliency_parameters", SALIENCY_PARAMS[saliency_active])


@change("input_file")
def process_file(input_file, image_url_1, image_url_2, image_count, **kwargs):
    """An image is getting loaded. Process the given image"""
    if not input_file:
        return

    # Make file available as image on HTML side
    _url = f"data:{input_file.get('type')};base64,{base64.encodebytes(input_file.get('content')).decode('utf-8')}"
    if not image_url_1 or image_count == 1:
        update_state("image_url_1", _url)
    elif not image_url_2 and image_count == 2:
        update_state("image_url_2", _url)


@change("image_url_1", "image_url_2")
def reset_image(image_url_1, image_url_2, image_count, **kwargs):
    """Method called when image_url_X is changed which can happen when setting them but also when cleared on the client side"""
    count = 0
    if image_url_1:
        count += 1
    if image_url_2 and image_count > 1:
        count += 1

    # Hide button if we have all the inputs we need
    update_state("need_input", count < image_count)


def run_model():
    """Method called when click prediction button"""
    print("Exec ML code for prediction")
    (image_url_1,) = get_state("image_url_1")
    update_state("predict_url", image_url_1)

    header, data = image_url_1.split(',')
    image = Image.open(io.BytesIO(base64.decodebytes(data.encode('utf-8'))))

    # TODO: Add model prediction (e.g. logits)
    model(model_loader(image))


def initialize(task_active, **kwargs):
    """Method called at startup time"""
    task_change(task_active)
    (saliency_active,) = get_state("saliency_active")
    saliency_change(saliency_active)


# Pytorch pre-processing
model_loader = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
