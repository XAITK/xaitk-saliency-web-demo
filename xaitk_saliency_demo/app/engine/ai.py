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

# labels
from .assets import imagenet_categories, coco_categories, coco_valid_idxs

# postprocess detections
from .postprocess import (
    faster_rcnn_postprocess_detections,
    retinanet_postprocess_detections,
)

from .singleton import Singleton

TOP_K = 5
FILL = np.uint8(np.asarray([0.485, 0.456, 0.406]) * 255)

AI_MAP = {
    "resnet-50": models.resnet50(pretrained=True),
    "alexnet": models.alexnet(pretrained=True),
    "vgg-16": models.vgg16(pretrained=True),
    "faster-rcnn": models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, box_score_thresh=0.0
    ),
    "retina-net": models.detection.retinanet_resnet50_fpn(
        pretrained=True, score_thresh=0.0, detections_per_img=100
    ),
}

XAI_MAP = {
    "RISEStack": RISEStack,
    "SlidingWindowStack": SlidingWindowStack,
    "similarity-saliency": {
        SlidingWindow: ["window_size", "stride"],
        SimilarityScoring: ["proximity_metric"],
    },
    "detection-saliency": {
        RISEGrid: ["n", "s", "p1", "seed", "threads"],
        DRISEScoring: ["proximity_metric"],
    },
}

# Pytorch pre-processing
imagenet_model_loader = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
coco_model_loader = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ]
)


# SMQTK black-box classifier
class ClfModel(ClassifyImage):
    def __init__(self, model, idx):
        self.model = model
        self.idx = idx

    def get_labels(self):
        return [imagenet_categories[i] for i in self.idx]

    @torch.no_grad()
    def classify_images(self, image_iter):
        for img in image_iter:
            inp = imagenet_model_loader(img).unsqueeze(0)
            vec = self.model(inp).cpu().numpy().squeeze()
            out = softmax(vec)
            yield dict(zip(self.get_labels(), out[self.idx]))

    def get_config(self):
        # Required by a parent class.
        return {}


@Singleton
class XaiController:
    def __init__(self):
        self._task = None
        self._model = None
        self._perturb = None
        self._saliency = None
        self._image_1 = None
        self._image_2 = None
        self._preds = None

    def can_run(self):
        return self._model is not None and self._image_1 is not None

    def set_task(self, task_name):
        self._task = task_name

    def set_model(self, model_name):
        try:
            self._model = AI_MAP[model_name]
            self._model.eval()
            # Perform monkey patch to get class probabilities
            if self._task == "detection":
                if model_name == "faster-rcnn":
                    RoIHeads.postprocess_detections = faster_rcnn_postprocess_detections
                else:
                    RetinaNet.postprocess_detections = retinanet_postprocess_detections
            # Perform model surgery to get feature descriptors
            elif self._task == "similarity":
                if model_name == "resnet-50":
                    self._model = nn.Sequential(*list(self._model.children())[:-1])
                else:
                    self._model.classifier = nn.Sequential(
                        *list(self._model.classifier.children())[:-1]
                    )
        except:
            print(f"Could not find {model_name} in {AI_MAP.keys()}")

    def set_saliency_method(self, method_name, params):
        try:
            if self._task == "detection" or self._task == "similarity":
                self._perturb, self._saliency = [
                    k(**{v: params[v] for v in v})
                    for (k, v) in XAI_MAP[method_name].items()
                ]
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
        if self._task == "detection":
            output = self._model(coco_model_loader(input).unsqueeze(0))[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"][:, coco_valid_idxs].cpu().numpy()
            output = format_detection(boxes, scores)  # .astype('float32')
        else:
            output = self._model(imagenet_model_loader(input).unsqueeze(0)).squeeze()
            output = output.cpu().numpy()

        return output

    def run_model(self):
        # Get model predictions
        preds = self.predict(self._image_1)
        output = None

        if self._task == "classification":
            preds = softmax(preds)
            topk = np.argsort(-preds)[:TOP_K]
            output = [(imagenet_categories[i], preds[i]) for i in topk]
            print(f"Predicted classes: {output}")
            self._preds = {
                "type": self._task,
                "topk": topk,
                "classes": output,
            }
        elif self._task == "detection":
            boxes = preds[:, :4]
            scores = np.max(preds[:, 5:], axis=1)
            scores_idx = np.argmax(preds[:, 5:], axis=1)
            topk = np.argsort(-scores)[:TOP_K]
            output = [
                (coco_categories[scores_idx[i]], scores[i], boxes[i]) for i in topk
            ]
            print(f"Predicted bounding boxes: {output}")
            self._preds = {
                "type": self._task,
                "topk": topk,
                "detection": output,
            }
        elif self._task == "similarity":
            preds_2 = self.predict(self._image_2)
            similarity = cosine_similarity(
                preds.reshape(1, -1), preds_2.reshape(1, -1)
            ).item()
            print(f"Similarity score: {similarity}")
            self._preds = {
                "type": self._task,
                "similarity": similarity,
            }
        return self._preds

    def run_saliency(self):
        output = {}

        if self._preds is None:
            return output

        if self._task == "detection":
            # generate reference prediction
            topk = self._preds["topk"]
            ref_preds = self.predict(self._image_1)[topk, :]
            # generate perturbed predictions
            pert_masks = self._perturb(self._image_1)
            pert_imgs = occlude_image_batch(self._image_1, pert_masks, FILL)
            pert_preds = np.asarray([self.predict(pi) for pi in pert_imgs])
            # generate saliency map
            sal = self._saliency(ref_preds, pert_preds, pert_masks)
            output = {
                "type": "detection",
                "references": ref_preds,
                "masks": pert_masks,
                "predictions": pert_preds,
                "saliency": sal,
            }
        elif self._task == "similarity":
            # generate query/reference features
            query_feat = self.predict(self._image_1)
            ref_feat = self.predict(self._image_2)
            # generate perturbed features
            pert_masks = self._perturb(self._image_2)
            pert_ref_imgs = occlude_image_batch(self._image_2, pert_masks, FILL)
            pert_ref_feats = np.asarray([self.predict(pi) for pi in pert_ref_imgs])
            # generate saliency map
            sal = self._saliency(query_feat, ref_feat, pert_ref_feats, pert_masks)
            output = {
                "type": "similarity",
                "references": ref_feat,
                "masks": pert_masks,
                "predictions": pert_ref_feats,
                "saliency": sal,
            }
        else:
            # classification
            topk = self._preds["topk"]
            self._saliency.fill = FILL
            sal = self._saliency(self._image_1, ClfModel(self._model, topk))
            output = {
                "type": "classification",
                "saliency": sal,
            }
        return output
