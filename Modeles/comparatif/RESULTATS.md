# Comparatif multi-modèles — détection EPI (helmet / head / safety-vest)

Conformément au §5.3 du sujet, trois architectures ont été entraînées et évaluées sur le
même dataset EPI (3 classes), avec les mêmes classes cibles et un protocole d'évaluation
identique (précision/rappel/F1/AP@50 à IoU≥0,5, latence GPU, nombre de paramètres).

## Protocole (identique pour les 3 modèles)

| | YOLO26-nano | Faster R-CNN (MobileNetV3-FPN) | SSDlite320 (MobileNetV3) |
|---|---|---|---|
| Famille | one-stage, temps réel | two-stage, plus précis mais plus lent | one-stage, très léger (edge) |
| Poids initiaux | pré-entraîné COCO | pré-entraîné COCO | pré-entraîné COCO |
| Images d'entraînement | 4 708 (dataset EPI complet, split train) | 4 708 | 4 708 |
| Epochs | 30 | 30 | 30 |
| Résolution | 640×640 | 640×640 | 640×640 |
| Matériel | GPU NVIDIA L4 (Scaleway, 24 Go VRAM) | idem | idem |

**Note méthodologique.** Les 3 modèles sont désormais entraînés avec un protocole
strictement identique (même dataset complet, même nombre d'epochs, même matériel GPU) —
une première version de ce comparatif avait été produite en CPU sur un sous-échantillon
(600 images, 3 epochs) faute de GPU disponible ; elle est remplacée par ces résultats,
obtenus après accès à un serveur GPU dédié (NVIDIA L4).

## Résultats (jeu de validation, 1 327 images)

| Modèle | Précision | Rappel | F1 | mAP@50 | Latence (ms/image, GPU) | Paramètres (M) | Taille poids |
|---|---|---|---|---|---|---|---|
| **YOLO26-nano** | 0,719 | 0,576 | 0,639* | 0,616 | ~68 | 2,50 | 5,3 Mo |
| **Faster R-CNN** (MobileNetV3-FPN) | 0,675 | 0,589 | 0,627 | 0,561 | ~11,5 | 18,94 | 76 Mo |
| **SSDlite320** (MobileNetV3) | 0,534 | 0,384 | 0,423 | 0,370 | ~4,9 | 2,23 | 9,2 Mo |

\* F1 recalculé à partir de precision/recall YOLO (2·P·R/(P+R)).

### Détail par classe (AP@50)

| Classe | Faster R-CNN | SSDlite320 |
|---|---|---|
| helmet | 0,532 | 0,239 |
| head | 0,843 | 0,687 |
| safety-vest | 0,309 | 0,185 |

_Le détail par classe pour YOLO (précision/rappel par classe, matrice de confusion) est
disponible dans `Modeles/runs/detect/yolo_epi_gpu/`._

## Analyse

- **YOLO26-nano** reste le meilleur compromis global (mAP@50 0,616), cohérent avec le
  choix retenu pour l'application (section 4.1 du rapport, argument temps réel) — même
  après avoir aligné le protocole d'entraînement des 3 modèles.
- **Faster R-CNN** progresse nettement par rapport à la première version du comparatif
  (mAP@50 0,561 contre 0,311 en CPU/600 images) et devient compétitif avec YOLO,
  notamment sur `head` (AP 0,843) — au prix d'un modèle 8× plus lourd (76 Mo vs 5-9 Mo).
- **SSDlite320** reste le plus léger et de très loin le plus rapide en inférence GPU
  (4,9 ms/image, 14× plus rapide que YOLO) : c'est le meilleur candidat pour un
  déploiement edge (caméra de chantier basse consommation), avec un rappel plus faible
  mais un `safety-vest` désormais détecté (AP 0,185, contre 0,000 en CPU/600 images).
- **`safety-vest` : le point le plus amélioré.** Sous-représentée dans le sous-échantillon
  de 600 images utilisé pour la première version du comparatif (d'où un AP à 0,000 pour
  les deux modèles torchvision), cette classe atteint désormais 0,309 (Faster R-CNN) et
  0,185 (SSDlite) sur le dataset complet — confirmation que le problème initial était bien
  un problème d'échantillon d'entraînement, pas d'architecture.

## Reproduire ces résultats

Dataset attendu dans `../dataset_yolo_epi` (voir `Donnees/README.md` pour le reconstruire) :

```
python3 comparatif/train_eval.py --model fasterrcnn --epochs 30 --train_size 999999 --val_size 999999 --img_size 640
python3 comparatif/train_eval.py --model ssdlite    --epochs 30 --train_size 999999 --val_size 999999 --img_size 640
```

Le device (CUDA/CPU) est détecté automatiquement. Sur GPU, l'entraînement complet prend
environ 1h40 pour Faster R-CNN et 50 min pour SSDlite (NVIDIA L4). Sur CPU, compter
plusieurs heures — réduire `--train_size`/`--epochs` pour un test rapide.

Les métriques JSON complètes sont dans `results/fasterrcnn_metrics.json` et
`results/ssdlite_metrics.json`. Les poids entraînés (non versionnés, ~76 Mo et ~9 Mo) sont
régénérés par ces commandes.
