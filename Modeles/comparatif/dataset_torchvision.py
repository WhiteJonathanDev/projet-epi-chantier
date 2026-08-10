"""Dataset EPI (3 classes: helmet, head, safety-vest) au format attendu par
torchvision.models.detection (labels YOLO -> boxes xyxy absolues + labels 1..3,
0 reserve au fond).
"""
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

CLASSES = ["__background__", "helmet", "head", "safety-vest"]


class EpiDataset(Dataset):
    def __init__(self, root, split, img_size=512, max_items=None, seed=0):
        self.img_size = img_size
        img_dir = Path(root) / "images" / split
        lbl_dir = Path(root) / "labels" / split

        all_imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        # On ne garde que les images qui ont au moins une annotation EPI (sinon rien a apprendre/evaluer)
        items = []
        for img_path in all_imgs:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists() and lbl_path.stat().st_size > 0:
                items.append((img_path, lbl_path))

        if max_items is not None and len(items) > max_items:
            rnd = random.Random(seed)
            items = rnd.sample(items, max_items)

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, lbl_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        w0, h0 = img.size
        img = img.resize((self.img_size, self.img_size))

        boxes, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, xc, yc, bw, bh = parts
                cls = int(cls)
                xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
                x1 = (xc - bw / 2) * self.img_size
                y1 = (yc - bh / 2) * self.img_size
                x2 = (xc + bw / 2) * self.img_size
                y2 = (yc + bh / 2) * self.img_size
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(self.img_size), x2), min(float(self.img_size), y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls + 1)  # decalage pour le fond=0

        image_tensor = F.to_tensor(img)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx]),
        }
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
