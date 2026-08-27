"""Tests unitaires des metriques de detection (Modeles/comparatif/metrics.py)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import average_precision_per_class, box_iou, precision_recall_f1


def test_box_iou_identical_boxes_is_one():
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    iou = box_iou(box, box)
    assert iou.shape == (1, 1)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0))


def test_box_iou_disjoint_boxes_is_zero():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    iou = box_iou(a, b)
    assert torch.isclose(iou[0, 0], torch.tensor(0.0))


def test_box_iou_half_overlap():
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b = torch.tensor([[5.0, 0.0, 15.0, 10.0]])
    # intersection = 5x10=50, union = 100+100-50=150 -> iou = 1/3
    iou = box_iou(a, b)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0 / 3.0), atol=1e-5)


def test_box_iou_empty_input_returns_empty_tensor():
    empty = torch.zeros((0, 4))
    box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    assert box_iou(empty, box).shape == (0, 1)
    assert box_iou(box, empty).shape == (1, 0)


def _make_sample(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
    pred = {
        "boxes": torch.tensor(pred_boxes, dtype=torch.float32) if pred_boxes else torch.zeros((0, 4)),
        "labels": torch.tensor(pred_labels, dtype=torch.int64) if pred_labels else torch.zeros((0,), dtype=torch.int64),
        "scores": torch.tensor(pred_scores, dtype=torch.float32) if pred_scores else torch.zeros((0,)),
    }
    target = {
        "boxes": torch.tensor(gt_boxes, dtype=torch.float32) if gt_boxes else torch.zeros((0, 4)),
        "labels": torch.tensor(gt_labels, dtype=torch.int64) if gt_labels else torch.zeros((0,), dtype=torch.int64),
    }
    return pred, target


def test_precision_recall_f1_perfect_match():
    pred, target = _make_sample(
        pred_boxes=[[0, 0, 10, 10]], pred_labels=[1], pred_scores=[0.9],
        gt_boxes=[[0, 0, 10, 10]], gt_labels=[1],
    )
    precision, recall, f1 = precision_recall_f1([pred], [target], num_classes=1)
    assert precision[0] == 1.0
    assert recall[0] == 1.0
    assert f1[0] == 1.0


def test_precision_recall_f1_false_positive_only():
    # Une prediction, aucune verite terrain -> precision 0, rappel indefini (0 par convention).
    pred, target = _make_sample(
        pred_boxes=[[0, 0, 10, 10]], pred_labels=[1], pred_scores=[0.9],
        gt_boxes=[], gt_labels=[],
    )
    precision, recall, f1 = precision_recall_f1([pred], [target], num_classes=1)
    assert precision[0] == 0.0
    assert recall[0] == 0.0


def test_precision_recall_f1_missed_detection():
    # Verite terrain presente, aucune prediction -> rappel 0.
    pred, target = _make_sample(
        pred_boxes=[], pred_labels=[], pred_scores=[],
        gt_boxes=[[0, 0, 10, 10]], gt_labels=[1],
    )
    precision, recall, f1 = precision_recall_f1([pred], [target], num_classes=1)
    assert recall[0] == 0.0


def test_precision_recall_f1_below_confidence_threshold_ignored():
    pred, target = _make_sample(
        pred_boxes=[[0, 0, 10, 10]], pred_labels=[1], pred_scores=[0.1],
        gt_boxes=[[0, 0, 10, 10]], gt_labels=[1],
    )
    precision, recall, f1 = precision_recall_f1([pred], [target], num_classes=1, conf_thresh=0.25)
    # La prediction est filtree par le seuil -> aucune detection -> rappel 0.
    assert recall[0] == 0.0


def test_average_precision_perfect_detector_is_one():
    pred, target = _make_sample(
        pred_boxes=[[0, 0, 10, 10]], pred_labels=[1], pred_scores=[0.99],
        gt_boxes=[[0, 0, 10, 10]], gt_labels=[1],
    )
    aps = average_precision_per_class([pred], [target], num_classes=1)
    assert aps[0] == pytest.approx(1.0, abs=1e-5)


def test_average_precision_no_ground_truth_is_zero():
    pred, target = _make_sample(
        pred_boxes=[[0, 0, 10, 10]], pred_labels=[1], pred_scores=[0.9],
        gt_boxes=[], gt_labels=[],
    )
    aps = average_precision_per_class([pred], [target], num_classes=1)
    assert aps[0] == 0.0
