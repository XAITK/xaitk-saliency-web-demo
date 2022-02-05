TASK_DEPENDENCY = {
    "similarity": {
        # Task => saliency
        "saliency_active": "similarity-saliency",
        "saliency_available": [
            {"text": "Default", "value": "similarity-saliency"},
        ],
        # Task => model
        "model_active": "SimilarityResNet50",
        "model_available": [
            {"text": "ResNet-50", "value": "SimilarityResNet50"},
            {"text": "AlexNet", "value": "SimilarityAlexNet"},
            {"text": "VGG-16", "value": "SimilarityVgg16"},
        ],
        # Task => input
        "input_expected": 2,
    },
    "detection": {
        # Task => saliency
        "saliency_active": "detection-saliency",
        "saliency_available": [
            {"text": "Default", "value": "detection-saliency"},
        ],
        # Task => model
        "model_active": "DetectionFasterRCNN",
        "model_available": [
            {"text": "Faster R-CNN", "value": "DetectionFasterRCNN"},
            {"text": "RetinaNet", "value": "DetectionRetinaNet"},
        ],
        # Task => input
        "input_expected": 1,
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
        "model_active": "ClassificationResNet50",
        "model_available": [
            {"text": "ResNet-50", "value": "ClassificationResNet50"},
            {"text": "AlexNet", "value": "ClassificationAlexNet"},
            {"text": "VGG-16", "value": "ClassificationVgg16"},
        ],
        # Task => input
        "input_expected": 1,
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
