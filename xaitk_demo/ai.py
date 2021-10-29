import io
import numpy as np
from PIL import Image
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity

# pytorch
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.retinanet import RetinaNet

# xaitk-saliency
from smqtk_classifier import ClassifyImage
from xaitk_saliency.utils.detection import format_detection
from xaitk_saliency.utils.masking import occlude_image_batch
from xaitk_saliency.impls.perturb_image.rise import RISEGrid
from xaitk_saliency.impls.perturb_image.sliding_window import SlidingWindow
from xaitk_saliency.impls.gen_detector_prop_sal.drise_scoring import DRISEScoring
from xaitk_saliency.impls.gen_descriptor_sim_sal.similarity_scoring import (
    SimilarityScoring,
)
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.rise import RISEStack
from xaitk_saliency.impls.gen_image_classifier_blackbox_sal.slidingwindow import (
    SlidingWindowStack,
)

# postprocess detections
from .postprocess import faster_rcnn_postprocess_detections, retinanet_postprocess_detections


TOP_K = 5
FILL = np.uint8(np.asarray([0.485, 0.456, 0.406]) * 255)

AI_MAP = {
    "resnet-50": models.resnet50(pretrained=True),
    "alexnet": models.alexnet(pretrained=True),
    "vgg-16": models.vgg16(pretrained=True),
    "faster-rcnn": models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                            box_score_thresh=0.0),
    "retina-net": models.detection.retinanet_resnet50_fpn(pretrained=True,
                                                          score_thresh=0.0,
                                                          detections_per_img=100),
}

XAI_MAP = {
    "RISEStack": RISEStack,
    "SlidingWindowStack": SlidingWindowStack,
    "similarity-saliency": {SlidingWindow: ["window_size", "stride"],
                            SimilarityScoring: ["proximity_metric"]},
    "detection-saliency": {RISEGrid: ["n", "s", "p1", "seed", "threads"],
                           DRISEScoring: ["proximity_metric"]},
}

# Pytorch pre-processing
imagenet_model_loader = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
)
coco_model_loader = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)

# Class labels associated with the ImageNet dataset
with open('data/imagenet_classes.txt') as f:
    imagenet_categories = f.read().splitlines()

# Class labels associated with the COCO dataset
# Source: https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection
coco_categories = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Filter out '__background__' and N/A' classes
valid_idxs = [idx for idx, x in enumerate(coco_categories) if x != 'N/A']
coco_categories = [coco_categories[idx] for idx in valid_idxs]


# SMQTK black-box classifier
# TODO: Test this is working...
class ClfModel(ClassifyImage):
    def __init__(self, model, idx):
        self.model = model
        self.idx = idx

    def get_labels(self):
        return ["output"]

    def classify_images(self, image_iter):
        for img in image_iter:
            inp = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
            out = self.model(inp)[0, self.idx].detach().numpy()
            yield dict(zip(self.get_labels(), [out]))

    def get_config(self):
        return {}


class XaiController:
    def __init__(self):
        self._task = None
        self._model = None
        self._perturb = None
        self._saliency = None
        self._image_1 = None
        self._image_2 = None
        self._saliency_params = {}

    def set_task(self, task_name):
        self._task = task_name

    def set_model(self, model_name):
        try:
            self._model = AI_MAP[model_name]
            self._model.eval()
            # Perform monkey patch to get class probabilities
            if self._task == 'detection':
                if model_name == 'faster-rcnn':
                    RoIHeads.postprocess_detections = faster_rcnn_postprocess_detections
                else:
                    RetinaNet.postprocess_detections = retinanet_postprocess_detections
            # Perform model surgery to get feature descriptors
            elif self._task == 'similarity':
                if model_name == 'resnet-50':
                    self._model = nn.Sequential(
                        *list(self._model.children())[:-1])
                else:
                    self._model.classifier = nn.Sequential(
                        *list(self._model.classifier.children())[:-1])
        except:
            print(f"Could not find {model_name} in {AI_MAP.keys()}")

    def set_saliency_method(self, method_name, params):
        try:
            if self._task == 'detection' or self._task == 'similarity':
                self._perturb, self._saliency = [
                    k(**{v: params[v] for v in v}) for (k, v) in XAI_MAP[method_name].items()]
            else:
                self._saliency = XAI_MAP[method_name](**params)
        except:
            print(f"Could not find {method_name} in {XAI_MAP.keys()}")

    def set_image_1(self, bytes_content):
        self._image_1 = np.array(Image.open(io.BytesIO(bytes_content)))

    def set_image_2(self, bytes_content):
        self._image_2 = np.array(Image.open(io.BytesIO(bytes_content)))

    @torch.no_grad()
    def predict(self, input):
        if self._task == 'detection':
            output = self._model(coco_model_loader(input).unsqueeze(0))[0]
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'][:, valid_idxs].cpu().numpy()
            output = format_detection(boxes, scores)
        else:
            output = self._model(imagenet_model_loader(
                input).unsqueeze(0)).squeeze()
            output = output.cpu().numpy()

        return output

    def run_model(self):
        # Get model predictions
        preds = self.predict(self._image_1)

        if self._task == 'classification':
            preds = softmax(preds)
            topk = np.argsort(-preds)[:TOP_K]
            output = [(imagenet_categories[i], preds[i]) for i in topk]
            print(f"Predicted classes: {output}")
        elif self._task == 'detection':
            boxes = preds[:, :4]
            scores = np.max(preds[:, 5:], axis=1)
            scores_idx = np.argmax(preds[:, 5:], axis=1)
            topk = np.argsort(-scores)[:TOP_K]
            output = [(coco_categories[scores_idx[i]], scores[i], boxes[i])
                      for i in topk]
            print(f"Predicted bounding boxes: {output}")
        elif self._task == 'similarity':
            preds_2 = self.predict(self._image_2)
            similarity = cosine_similarity(preds.reshape(
                1, -1), preds_2.reshape(1, -1)).item()
            print(f"Similarity score: {similarity}")

    def run_saliency(self):
        if self._task == 'detection':
            # generate reference prediction
            ref_preds = self.predict(self._image_1)
            # generate perturbed predictions
            pert_masks = self._perturb(self._image_1)
            pert_imgs = occlude_image_batch(self._image_1, pert_masks, FILL)
            pert_preds = np.asarray([self.predict(pi) for pi in pert_imgs])
            # generate saliency map
            sal = self._saliency(ref_preds, pert_preds, pert_masks)
        elif self._task == 'similarity':
            # generate query/reference features
            query_feat = self.predict(self._image_1)
            ref_feat = self.predict(self._image_2)
            # generate perturbed features
            pert_masks = self._perturb(self._image_2)
            pert_ref_imgs = occlude_image_batch(
                self._image_2, pert_masks, FILL)
            pert_ref_feats = np.asarray(
                [self.predict(pi) for pi in pert_ref_imgs])
            # generate saliency map
            sal = self._saliency(query_feat, ref_feat,
                                 pert_ref_feats, pert_masks)
        else:
            # classification
            sal = self._saliency(self._image_1, ClfModel(self._model))

    def update_saliency_params(self, **kwargs):
        self._saliency_params.update(kwargs)
