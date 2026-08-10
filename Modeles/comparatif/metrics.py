"""Metriques de detection : precision/rappel/F1 a un seuil de confiance donne,
et AP@50 par classe (average precision, IoU>=0.5), pour comparer les modeles
entre eux avec les memes indicateurs que la partie YOLO du projet.
"""
import numpy as np
import torch


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-8)


def precision_recall_f1(all_preds, all_targets, num_classes, conf_thresh=0.25, iou_thresh=0.5):
    """all_preds/all_targets: listes paralleles de dicts {boxes, labels, scores?}."""
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for pred, target in zip(all_preds, all_targets):
        keep = pred["scores"] >= conf_thresh
        p_boxes = pred["boxes"][keep]
        p_labels = pred["labels"][keep]

        t_boxes = target["boxes"]
        t_labels = target["labels"]

        matched_gt = set()
        for cls in range(1, num_classes + 1):
            p_idx = (p_labels == cls).nonzero(as_tuple=True)[0]
            t_idx = (t_labels == cls).nonzero(as_tuple=True)[0]
            if len(p_idx) == 0 and len(t_idx) == 0:
                continue
            ious = box_iou(p_boxes[p_idx], t_boxes[t_idx]) if len(p_idx) and len(t_idx) else torch.zeros((len(p_idx), len(t_idx)))
            used_t = set()
            for i in range(len(p_idx)):
                if len(t_idx) == 0:
                    fp[cls - 1] += 1
                    continue
                best_j = int(torch.argmax(ious[i])) if ious.numel() else -1
                if best_j >= 0 and ious[i, best_j] >= iou_thresh and best_j not in used_t:
                    tp[cls - 1] += 1
                    used_t.add(best_j)
                else:
                    fp[cls - 1] += 1
            fn[cls - 1] += len(t_idx) - len(used_t)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1


def average_precision_per_class(all_preds, all_targets, num_classes, iou_thresh=0.5):
    """AP@50 par classe (methode standard : tri par score decroissant, aire sous la courbe PR)."""
    aps = np.zeros(num_classes)

    for cls in range(1, num_classes + 1):
        detections = []  # (score, image_idx, box)
        n_gt = 0
        gt_by_image = {}
        for img_idx, (pred, target) in enumerate(zip(all_preds, all_targets)):
            t_idx = (target["labels"] == cls).nonzero(as_tuple=True)[0]
            gt_by_image[img_idx] = target["boxes"][t_idx]
            n_gt += len(t_idx)

            p_idx = (pred["labels"] == cls).nonzero(as_tuple=True)[0]
            for i in p_idx:
                detections.append((float(pred["scores"][i]), img_idx, pred["boxes"][i]))

        if n_gt == 0:
            aps[cls - 1] = 0.0
            continue

        detections.sort(key=lambda x: -x[0])
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        matched = {img_idx: set() for img_idx in gt_by_image}

        for i, (score, img_idx, box) in enumerate(detections):
            gts = gt_by_image[img_idx]
            if gts.numel() == 0:
                fp[i] = 1
                continue
            ious = box_iou(box.unsqueeze(0), gts)[0]
            best_j = int(torch.argmax(ious))
            if ious[best_j] >= iou_thresh and best_j not in matched[img_idx]:
                tp[i] = 1
                matched[img_idx].add(best_j)
            else:
                fp[i] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / n_gt
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

        # aire sous la courbe PR (all-point interpolation)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
        aps[cls - 1] = ap

    return aps
