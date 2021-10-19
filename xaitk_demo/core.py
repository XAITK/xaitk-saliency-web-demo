import base64
from trame import change, get_state, update_state
from .ai import XaiController

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
ALL_SALIENCY_PARAMS = [
    "window_size",
    "stride",
    "similarity_metric",
    "n",
    "s",
    "p1",
    "seed",
    "threads",
    "proximity_metric",
    "debiased",
]

XAI = XaiController()


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
    XAI.set_model(model_active)


@change("saliency_active")
def saliency_change(saliency_active, **kwargs):
    """Saliency algo is changing"""
    print("Use saliency", saliency_active)
    update_state("saliency_parameters", SALIENCY_PARAMS[saliency_active])
    XAI.set_saliency_method(saliency_active)


@change("input_file")
def process_file(input_file, image_url_1, image_url_2, image_count, **kwargs):
    """An image is getting loaded. Process the given image"""
    if not input_file:
        return

    # Make file available as image on HTML side
    _url = f"data:{input_file.get('type')};base64,{base64.encodebytes(input_file.get('content')).decode('utf-8')}"
    if not image_url_1 or image_count == 1:
        update_state("image_url_1", _url)
        XAI.set_image_1(input_file.get("content"))
    elif not image_url_2 and image_count == 2:
        update_state("image_url_2", _url)
        XAI.set_image_2(input_file.get("content"))


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


@change(*ALL_SALIENCY_PARAMS)
def saliency_param_update(**kwargs):
    params = {}
    for name in ALL_SALIENCY_PARAMS:
        params[name] = kwargs.get(name)
    XAI.update_saliency_params(**params)


def run_model():
    """Method called when click prediction button"""
    print("Exec ML code for prediction")
    (image_url_1,) = get_state("image_url_1")
    update_state("predict_url", image_url_1)
    XAI.run_model()


def initialize(task_active, **kwargs):
    """Method called at startup time"""
    task_change(task_active)
    (saliency_active,) = get_state("saliency_active")
    saliency_change(saliency_active)
