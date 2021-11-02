from typing import Dict, List, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops


def faster_rcnn_postprocess_detections(
    self,
    class_logits,  # type: Tensor
    box_regression,  # type: Tensor
    proposals,  # type: List[Tensor]
    # type: List[Tuple[int, int]]
    image_shapes,
):
    # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = self.box_coder.decode(box_regression, proposals)

    pred_scores = F.softmax(class_logits, -1)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes = []
    all_scores = []
    all_labels = []
    for boxes, scores, image_shape in zip(
        pred_boxes_list, pred_scores_list, image_shapes
    ):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        # create labels for each prediction
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        # remove predictions with the background label
        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        scores_orig = scores.clone()
        labels = labels[:, 1:]

        # batch everything, by making every class prediction be a separate instance
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        # remove low scoring boxes
        inds = torch.where(scores > self.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        # remove empty boxes
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        inds = inds[keep]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # non-maximum suppression, independently done per class
        keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
        # keep only topk scoring predictions
        keep = keep[: self.detections_per_img]
        inds = inds[keep]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # Find corresponding row of matrix
        inds = inds // (num_classes - 1)

        all_boxes.append(boxes)
        all_scores.append(scores_orig[inds, :])
        all_labels.append(labels)

    return all_boxes, all_scores, all_labels


def retinanet_postprocess_detections(self, head_outputs, anchors, image_shapes):
    # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
    class_logits = head_outputs["cls_logits"]
    box_regression = head_outputs["bbox_regression"]

    num_images = len(image_shapes)

    detections: List[Dict[str, Tensor]] = []

    for index in range(num_images):
        box_regression_per_image = [br[index] for br in box_regression]
        logits_per_image = [cl[index] for cl in class_logits]
        anchors_per_image, image_shape = anchors[index], image_shapes[index]

        image_boxes = []
        image_scores = []
        image_labels = []
        image_anchors = []

        for box_regression_per_level, logits_per_level, anchors_per_level in zip(
            box_regression_per_image, logits_per_image, anchors_per_image
        ):
            num_classes = logits_per_level.shape[-1]

            # remove low scoring boxes
            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            keep_idxs = scores_per_level > self.score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions
            num_topk = min(self.topk_candidates, topk_idxs.size(0))
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            labels_per_level = topk_idxs % num_classes

            boxes_per_level = self.box_coder.decode_single(
                box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
            )
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)
            image_anchors.append(anchor_idxs)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_anchors = torch.cat(image_anchors, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(
            image_boxes, image_scores, image_labels, self.nms_thresh
        )
        keep = keep[: self.detections_per_img]

        # Recover original class logits
        level = torch.div(keep, num_topk, rounding_mode="floor")
        image_logits = torch.sigmoid(
            torch.stack(
                [logits_per_image[l][a] for (l, a) in zip(level, image_anchors[keep])]
            )
        )

        detections.append(
            {
                "boxes": image_boxes[keep],
                # remove background class
                "scores": image_logits[:, 1:],
                "labels": image_labels[keep],
            }
        )

    return detections
