import base64
from trame import change, get_state, update_state
from xaitk_demo.ai import XaiController

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
        "saliency_active": "similarity-saliency",
        "saliency_available": [
            {"text": "Default", "value": "similarity-saliency"},
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
        "saliency_active": "detection-saliency",
        "saliency_available": [
            {"text": "Default", "value": "detection-saliency"},
        ],
        # Task => model
        "model_active": "faster-rcnn",
        "model_available": [
            {"text": "Faster R-CNN", "value": "faster-rcnn"},
            {"text": "RetinaNet", "value": "retina-net"},
        ],
        # Task => input
        "image_count": 1,
        # Better defaults:
        "n": 200,
        "proximity_metric": "cosine",
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
    "RISEStack": ["n", "s", "p1", "seed", "threads", "debiased"],
    "SlidingWindowStack": ["window_size", "stride", "threads"],
    "similarity-saliency": ["window_size", "stride", "proximity_metric"],
    "detection-saliency": ["n", "s", "p1", "seed", "threads", "proximity_metric"],
}

ALL_SALIENCY_PARAMS = {
    "window_size": int,
    "stride": int,
    "n": int,
    "s": int,
    "p1": float,
    "seed": int,
    "threads": int,
    "proximity_metric": str,
    "debiased": bool,
}

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
    reset_xai_viz()

    print("Use task", task_active)
    XAI.set_task(task_active)


@change("model_active")
def model_change(model_active, **kwargs):
    """ML model is changing"""
    print("Use model", model_active)
    XAI.set_model(model_active)
    reset_xai_viz()


@change("saliency_active")
def saliency_change(saliency_active, **kwargs):
    """Saliency algo is changing"""
    print("Use saliency", saliency_active)
    update_state("saliency_parameters", SALIENCY_PARAMS[saliency_active])
    params = {}
    for name in SALIENCY_PARAMS[saliency_active]:
        value = kwargs.get(name)
        convert = ALL_SALIENCY_PARAMS[name]
        if isinstance(value, list):
            params[name] = [convert(v) for v in value]
        else:
            params[name] = convert(value)

    XAI.set_saliency_method(saliency_active, params)
    reset_xai_viz()


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
        if image_count == 1:
            run_model()
    elif not image_url_2 and image_count == 2:
        update_state("image_url_2", _url)
        XAI.set_image_2(input_file.get("content"))
        run_model()

    reset_xai_viz()


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
    reset_xai_viz()


@change(*list(ALL_SALIENCY_PARAMS.keys()))
def saliency_param_update(**kwargs):
    print("Updating saliency params")
    params = {}
    for name in ALL_SALIENCY_PARAMS:
        value = kwargs.get(name)
        convert = ALL_SALIENCY_PARAMS[name]
        if isinstance(value, list):
            params[name] = [convert(v) for v in value]
        else:
            params[name] = convert(value)

    XAI.update_saliency_params(**params)
    (saliency_active,) = get_state("saliency_active")
    saliency_change(saliency_active, **params)
    reset_xai_viz()


def run_model():
    """Method called when click prediction button"""
    print("Exec ML code for prediction")
    from .ui import update_prediction

    update_prediction(XAI.run_model())


def run_saliency():
    """Method called when click saliency button"""
    print("Exec saliency code for explanation")
    output = XAI.run_saliency()
    print("run_saliency")
    update_state("xai_type", output.get("type"))

    if output.get("type") == "classification":
        _xai_saliency = output.get("saliency")
        nb_classes = _xai_saliency.shape[0]
        heat_maps = {}
        for i in range(nb_classes):
            _key = f"heatmap_{i}"
            heat_maps[_key] = _xai_saliency[i].ravel().tolist()

        update_state("xai_class_heatmaps", heat_maps)

    elif output.get("type") == "similarity":
        _xai_saliency = output.get("saliency")
        heat_maps = {
            "heatmap_0": _xai_saliency.ravel().tolist(),
        }
        update_state("xai_similarity_heatmaps", heat_maps)
    elif output.get("type") == "detection":
        _xai_saliency = output.get("saliency")
        nb_classes = _xai_saliency.shape[0]
        heat_maps = {}
        for i in range(nb_classes):
            _key = f"heatmap_{i}"
            heat_maps[_key] = _xai_saliency[i].ravel().tolist()

        update_state("xai_detection_heatmaps", heat_maps)

    else:
        print(output.get("type"))
        for key, value in output.items():
            if key != "type":
                print(f"{key}: {value.shape} | {value.dtype}")


def initialize(task_active, **kwargs):
    """Method called at startup time"""
    task_change(task_active)
    (model_active,) = get_state("model_active")
    model_change(model_active)
    saliency_param_update(**kwargs)
    reset_xai_viz()


def reset_xai_viz():
    update_state("xai_type", None)


@change("heatmap_color_min", "heatmap_color_max")
def heatmap_color_min_change(heatmap_color_min, heatmap_color_max, **kwargs):
    try:
        update_state(
            "xai_color_range", [float(heatmap_color_min), float(heatmap_color_max)]
        )
    except:
        pass
