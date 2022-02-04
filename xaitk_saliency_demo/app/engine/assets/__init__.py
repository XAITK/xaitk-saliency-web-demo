from pathlib import Path

BASE_PATH = str(Path(__file__).parent.absolute())

with open(Path(BASE_PATH, "imagenet_classes.txt")) as f:
    imagenet_categories = f.read().splitlines()

# Class labels associated with the COCO dataset
# Source: https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
with open(Path(BASE_PATH, "coco_classes.txt")) as f:
    coco_categories = f.read().splitlines()

    # Filter out '__background__' and N/A' classes
    coco_valid_idxs = [idx for idx, x in enumerate(coco_categories) if x != "N/A"]
    coco_categories = [coco_categories[idx] for idx in coco_valid_idxs]
