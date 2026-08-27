"""Tests unitaires du dataset EPI (Modeles/comparatif/dataset_torchvision.py) :
conversion des labels YOLO (normalises, xc/yc/w/h) vers boxes absolues xyxy.
"""
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_torchvision import EpiDataset


def _make_mini_dataset(tmp_path, img_size_px=100):
    root = tmp_path / "dataset_yolo_epi"
    for kind in ("images", "labels"):
        (root / kind / "train").mkdir(parents=True)

    img = Image.new("RGB", (img_size_px, img_size_px), color="white")
    img.save(root / "images" / "train" / "sample.jpg")

    # Une bbox centree (xc=0.5, yc=0.5), largeur/hauteur = 0.2 -> attendu en
    # pixels sur une image redimensionnee a IMG_SIZE : [0.4*S, 0.4*S, 0.6*S, 0.6*S]
    (root / "labels" / "train" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    # Une deuxieme image sans annotation (ne doit pas etre incluse par EpiDataset,
    # qui filtre les images sans fichier label non vide).
    Image.new("RGB", (img_size_px, img_size_px)).save(root / "images" / "train" / "empty.jpg")
    (root / "labels" / "train" / "empty.txt").write_text("")

    return root


def test_epidataset_only_keeps_annotated_images(tmp_path):
    root = _make_mini_dataset(tmp_path)
    ds = EpiDataset(str(root), "train", img_size=64)
    assert len(ds) == 1  # "empty.jpg" est exclue


def test_epidataset_converts_yolo_coords_to_absolute_xyxy(tmp_path):
    root = _make_mini_dataset(tmp_path)
    img_size = 64
    ds = EpiDataset(str(root), "train", img_size=img_size)

    image_tensor, target = ds[0]
    assert image_tensor.shape == (3, img_size, img_size)

    boxes = target["boxes"]
    labels = target["labels"]
    assert boxes.shape == (1, 4)
    # EpiDataset decale les labels de +1 (0 = fond, reserve par torchvision) :
    # la classe YOLO 0 (helmet) devient le label torchvision 1.
    assert labels.tolist() == [1]

    expected = img_size * 0.4, img_size * 0.4, img_size * 0.6, img_size * 0.6
    for got, exp in zip(boxes[0].tolist(), expected):
        assert got == pytest.approx(exp, abs=0.5)


def test_epidataset_max_items_caps_dataset_size(tmp_path):
    root = tmp_path / "dataset_yolo_epi"
    for kind in ("images", "labels"):
        (root / kind / "train").mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (20, 20)).save(root / "images" / "train" / f"img{i}.jpg")
        (root / "labels" / "train" / f"img{i}.txt").write_text("1 0.5 0.5 0.1 0.1\n")

    ds = EpiDataset(str(root), "train", img_size=32, max_items=2, seed=0)
    assert len(ds) == 2
