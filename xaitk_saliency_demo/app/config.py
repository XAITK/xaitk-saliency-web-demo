TASK_DEPENDENCY = {
    "similarity": {
        # Task => saliency
        "saliency_active": "SBSMStack",
        "saliency_available": [
            {"title": "SBSM", "value": "SBSMStack"},
        ],
        # Task => model
        "model_active": "SimilarityResNet50",
        "model_available": [
            {"title": "ResNet-50", "value": "SimilarityResNet50"},
            {"title": "AlexNet", "value": "SimilarityAlexNet"},
            {"title": "VGG-16", "value": "SimilarityVgg16"},
        ],
        # Task => input
        "input_expected": 2,
    },
    "detection": {
        # Task => saliency
        "saliency_active": "DRISEStack",
        "saliency_available": [
            {"title": "DRISE", "value": "DRISEStack"},
            {"title": "RandomGridStack", "value": "RandomGridStack"},
        ],
        # Task => model
        "model_active": "DetectionFRCNN",
        "model_available": [
            {"title": "FRCNN (COCO)", "value": "DetectionFRCNN"},
            {"title": "CenterNet (VisDrone)", "value": "DetectionCenterNetVisdrone"},
        ],
        # Task => input
        "input_expected": 1,
    },
    "classification": {
        # Task => saliency
        "saliency_active": "RISEStack",
        "saliency_available": [
            {"title": "RISE Stack", "value": "RISEStack"},
            {"title": "Sliding Window Stack", "value": "SlidingWindowStack"},
        ],
        # Task => model
        "model_active": "ClassificationResNet50",
        "model_available": [
            {"title": "ResNet-50", "value": "ClassificationResNet50"},
            {"title": "AlexNet", "value": "ClassificationAlexNet"},
            {"title": "VGG-16", "value": "ClassificationVgg16"},
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
    "RandomGridStack": ["n", "s_tuple", "p1", "seed", "threads"],
    # classification
    "RISEStack": ["n", "s", "p1", "seed", "threads", "debiased"],
    "SlidingWindowStack": ["window_size", "stride", "threads"],
}

ALL_SALIENCY_PARAMS = {
    "window_size": int,
    "stride": int,
    "n": int,
    "s": int,
    "s_tuple": int,
    "p1": float,
    "seed": int,
    "threads": int,
    "proximity_metric": str,
    "debiased": bool,
}

SALINECY_PARAM_REMAP = {"s": "s_tuple"}

TASKS = [
    {"title": "Image Similarity", "value": "similarity"},
    {"title": "Object Detection", "value": "detection"},
    {"title": "Image Classification", "value": "classification"},
]

PROXIMITY_METRIC_AVAILABLE = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "cityblock",
    "correlation",
    "cosine",
    "dice",
    "euclidean",
    "hamming",
    "jaccard",
    "jensenshannon",
    "kulsinski",
    "mahalanobis",
    "matching",
    "minkowski",
    "rogerstanimoto",
    "russellrao",
    "seuclidean",
    "sokalmichener",
    "sokalsneath",
    "sqeuclidean",
    "wminkowski",
    "yule",
]

# https://github.com/Kitware/trame-components/blob/master/vue-components/src/components/XaiHeatMap/script.js#L47-L78
HEAT_MAP_MODES = [
    ("full", "mdi-arrow-left-right", "true"),
    ("maxSym", "mdi-arrow-expand-horizontal", "true"),
    ("minSym", "mdi-arrow-collapse-horizontal", "full_range[0] < 0"),
    ("negative", "mdi-ray-end-arrow", "full_range[0] < 0"),
    ("positive", "mdi-ray-start-arrow", "full_range[1] > 0"),
    ("custom", "mdi-account", "true"),
]

# Common css style
STYLE_COMPACT = dict(hide_details=True, density="compact", variant="underlined")
STYLE_SELECT = dict(style="max-width: 200px", classes="mx-2")
STYLE_ROW = dict(style="min-width: 0;", classes="d-flex flex-shrink-1")
