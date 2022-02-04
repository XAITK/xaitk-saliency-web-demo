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
