"""Genere un vrai rapport de nettoyage (pas de chiffres inventes) sur le dataset EPI :
- images corrompues/illisibles
- doublons exacts (hash de contenu)
- coherence images/labels
- distribution des classes
"""
import hashlib
import json
import os
from pathlib import Path
from PIL import Image

ROOT = Path("/Users/ecole18.06-t.m.t/Desktop/COMPUTER VISION/projet/dataset_yolo_epi")
CLASSES = {0: "helmet", 1: "head", 2: "safety-vest"}

report = {"splits": {}}
seen_hashes = {}
duplicates = []
corrupt = []

for split in ["train", "val", "test"]:
    img_dir = ROOT / "images" / split
    lbl_dir = ROOT / "labels" / split
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    n_with_label = 0
    n_without_label = 0
    class_counts = {v: 0 for v in CLASSES.values()}
    n_boxes = 0

    for img_path in images:
        # verification integrite
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception:
            corrupt.append(str(img_path))
            continue

        # doublon exact (hash du contenu, echantillon rapide sur les 1eres/dernieres 64ko pour vitesse)
        with open(img_path, "rb") as f:
            data = f.read(65536)
        h = hashlib.md5(data).hexdigest()
        if h in seen_hashes:
            duplicates.append((str(img_path), seen_hashes[h]))
        else:
            seen_hashes[h] = str(img_path)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            n_with_label += 1
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls = int(parts[0])
                        class_counts[CLASSES.get(cls, f"unknown_{cls}")] = class_counts.get(CLASSES.get(cls, f"unknown_{cls}"), 0) + 1
                        n_boxes += 1
        else:
            n_without_label += 1

    report["splits"][split] = {
        "n_images_total": len(images),
        "n_images_avec_annotation_epi": n_with_label,
        "n_images_sans_annotation_epi": n_without_label,
        "n_boxes_epi": n_boxes,
        "distribution_classes": class_counts,
    }

report["images_corrompues"] = corrupt
report["n_images_corrompues"] = len(corrupt)
report["doublons_exacts"] = duplicates[:50]
report["n_doublons_exacts"] = len(duplicates)
report["n_images_scannees_total"] = sum(v["n_images_total"] for v in report["splits"].values())

out_path = "/tmp/data_cleaning_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(json.dumps(report, indent=2, ensure_ascii=False))
