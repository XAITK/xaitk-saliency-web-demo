TASK_DEPENDENCY = {
    "similarity": {
        # Task => saliency
        "saliency_active": "SBSMStack",
        "saliency_available": [
            {"text": "SBSM", "value": "SBSMStack"},
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
        "saliency_active": "DRISEStack",
        "saliency_available": [
            {"text": "DRISE", "value": "DRISEStack"},
            {"text": "RandomGridStack", "value": "RandomGridStack"}
        ],
        # Task => model
        "model_active": "DetectionFRCNN",
        "model_available": [
            {"text": "FRCNN (COCO)", "value": "DetectionFRCNN"},
            {"text": "CenterNet (VisDrone)", "value": "DetectionCenterNetVisdrone"},
        ],
        # Task => input
        "input_expected": 1,
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
    # similarity
    "SBSMStack": ["window_size", "stride", "proximity_metric", "threads"],
    # detection
    "DRISEStack": ["n", "s", "p1", "seed", "threads"],
    "RandomGridStack": ["n", "s", "p1", "seed", "threads"],
    # classification
    "RISEStack": ["n", "s", "p1", "seed", "threads", "debiased"],
    "SlidingWindowStack": ["window_size", "stride", "threads"],
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
