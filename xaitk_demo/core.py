from trame import change, get_state, update_state

import base64

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
        "model_active": "pytorch-a",
        "model_available": [
            {"text": "PyTorch A", "value": "pytorch-a"},
            {"text": "PyTorch B", "value": "pytorch-b"},
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
        "model_active": "pytorch-a",
        "model_available": [
            {"text": "PyTorch A", "value": "pytorch-a"},
            {"text": "PyTorch B", "value": "pytorch-b"},
        ],
        # Task => input
        "image_count": 1,
    },
}

SALIENCY_PARAMS = {
    "SlidingWindow": ["window_size", "stride"],
    "SimilarityScoring": ["window_size", "stride", "similarity_metric"],
    "RISEGrid": ["n", "s", "p1", "proximity_metric", "seed", "threads"],
    "DRISEScoring": ["n", "s", "p1", "proximity_metric", "seed", "threads"],
    "RISEStack": ["n", "s", "p1", "proximity_metric", "seed", "threads"],
    "SlidingWindowStack": ["n", "s", "p1", "proximity_metric", "seed", "threads"],
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
    print(f"Use model {model_active}")


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


def initialize(task_active, **kwargs):
    """Method called at startup time"""
    task_change(task_active)
    (saliency_active,) = get_state("saliency_active")
    saliency_change(saliency_active)
