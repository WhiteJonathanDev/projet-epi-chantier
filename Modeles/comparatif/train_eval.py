"""Entraine (fine-tuning, tete de detection reentrainee) et evalue un modele
torchvision sur le dataset EPI (3 classes), pour comparaison avec YOLO.

Usage (dataset attendu dans ../dataset_yolo_epi par defaut, cf. Donnees/README.md pour
le reconstruire) :
    python3 train_eval.py --model fasterrcnn --epochs 30 --train_size 999999 --val_size 999999 --img_size 640
    python3 train_eval.py --model ssdlite --epochs 30 --train_size 999999 --val_size 999999 --img_size 640

Resultats GPU (NVIDIA L4, dataset complet, 30 epochs) dans results/*.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead

from dataset_torchvision import EpiDataset, collate_fn, CLASSES
from metrics import precision_recall_f1, average_precision_per_class

NUM_CLASSES = len(CLASSES)  # background + 3
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = os.environ.get("EPI_DATASET_ROOT", str(SCRIPT_DIR.parent.parent / "dataset_yolo_epi"))
OUT_DIR = Path(os.environ.get("EPI_RESULTS_DIR", str(SCRIPT_DIR / "results")))
OUT_DIR.mkdir(exist_ok=True, parents=True)


def build_model(name):
    if name == "fasterrcnn":
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        return model
    elif name == "ssdlite":
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        in_channels = [c[0][0].in_channels for c in model.head.classification_head.module_list]
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, NUM_CLASSES, norm_layer=torch.nn.BatchNorm2d
        )
        return model
    raise ValueError(name)


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    # NB: le backend MPS s'est avere instable (blocage observe) pour l'entrainement
    # de fasterrcnn/ssdlite sur cette machine -> CPU par defaut, plus lent mais fiable.
    return torch.device("cpu")


def train(model, loader, device, epochs, lr=0.001):
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for images, targets in loader:
            tb0 = time.time()
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            if n_batches % 10 == 0:
                print(f"  epoch {epoch+1} batch {n_batches}: {time.time()-tb0:.2f}s/batch, "
                      f"loss={loss.item():.3f}, elapsed={time.time()-t0:.0f}s", flush=True)
        print(f"[epoch {epoch+1}/{epochs}] loss={epoch_loss/max(n_batches,1):.4f} "
              f"({time.time()-t0:.1f}s, {n_batches} batches)", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, conf_thresh=0.25):
    model.to(device)
    model.eval()
    all_preds, all_targets = [], []
    latencies = []
    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        t0 = time.time()
        outputs = model(images_dev)
        if device.type == "mps":
            torch.mps.synchronize()
        latencies.append((time.time() - t0) / len(images))
        for out, tgt in zip(outputs, targets):
            all_preds.append({k: v.detach().cpu() for k, v in out.items()})
            all_targets.append({k: v.detach().cpu() for k, v in tgt.items()})

    precision, recall, f1 = precision_recall_f1(all_preds, all_targets, NUM_CLASSES - 1, conf_thresh=conf_thresh)
    aps = average_precision_per_class(all_preds, all_targets, NUM_CLASSES - 1)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "precision_per_class": dict(zip(CLASSES[1:], precision.tolist())),
        "recall_per_class": dict(zip(CLASSES[1:], recall.tolist())),
        "f1_per_class": dict(zip(CLASSES[1:], f1.tolist())),
        "ap50_per_class": dict(zip(CLASSES[1:], aps.tolist())),
        "map50": float(aps.mean()),
        "precision_mean": float(precision.mean()),
        "recall_mean": float(recall.mean()),
        "f1_mean": float(f1.mean()),
        "latency_ms_per_image": float(sum(latencies) / len(latencies) * 1000),
        "n_params_millions": n_params / 1e6,
        "device": device.type,
        "n_eval_images": len(all_preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["fasterrcnn", "ssdlite"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_size", type=int, default=1200)
    parser.add_argument("--val_size", type=int, default=300)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = get_device(force_cpu=args.cpu)
    print(f"Device: {device}", flush=True)

    train_ds = EpiDataset(DATASET_ROOT, "train", img_size=args.img_size, max_items=args.train_size)
    val_ds = EpiDataset(DATASET_ROOT, "val", img_size=args.img_size, max_items=args.val_size)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=min(8, os.cpu_count() or 4))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=min(8, os.cpu_count() or 4))

    model = build_model(args.model)

    t0 = time.time()
    train(model, train_loader, device, args.epochs, lr=args.lr)
    train_time = time.time() - t0

    metrics = evaluate(model, val_loader, device)
    metrics["train_time_seconds"] = train_time
    metrics["model"] = args.model
    metrics["epochs"] = args.epochs
    metrics["train_size"] = len(train_ds)
    metrics["val_size"] = len(val_ds)
    metrics["img_size"] = args.img_size

    out_json = OUT_DIR / f"{args.model}_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Resultats sauvegardes: {out_json}", flush=True)

    weights_path = OUT_DIR / f"{args.model}_weights.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"Poids sauvegardes: {weights_path}", flush=True)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
