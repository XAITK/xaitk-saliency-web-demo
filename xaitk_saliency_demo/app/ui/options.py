TASKS = [
    {"text": "Image Similarity", "value": "similarity"},
    {"text": "Object Detection", "value": "detection"},
    {"text": "Image Classification", "value": "classification"},
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
