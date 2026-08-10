# Comparatif multi-modèles — détection EPI (helmet / head / safety-vest)

Conformément au §5.3 du sujet, trois architectures ont été entraînées et évaluées sur le
même dataset EPI (3 classes), avec les mêmes classes cibles et un protocole d'évaluation
identique (précision/rappel/F1/AP@50 à IoU≥0,5, latence CPU, nombre de paramètres).

## Protocole

| | YOLO26-nano | Faster R-CNN (MobileNetV3-FPN) | SSDlite320 (MobileNetV3) |
|---|---|---|---|
| Famille | one-stage, temps réel | two-stage, plus précis mais plus lent | one-stage, très léger (edge) |
| Poids initiaux | pré-entraîné COCO | pré-entraîné COCO | pré-entraîné COCO |
| Images d'entraînement | 5 832 (dataset EPI complet) | 600 (sous-échantillon) | 600 (sous-échantillon) |
| Epochs | 20 | 3 | 3 |
| Résolution | 640×640 | 320×320 | 320×320 |
| Matériel | CPU (Apple Silicon) | CPU (Apple Silicon) | CPU (Apple Silicon) |

**Note méthodologique.** Faster R-CNN et SSDlite sont entraînés sur un sous-échantillon
(600 images) et moins d'epochs que YOLO, faute de ressources GPU dédiées — un compromis
temps/qualité assumé (cf. Rapport, section 4.2, qui documente le même arbitrage pour
YOLO). Le classement relatif entre architectures reste néanmoins informatif ; un
passage à l'échelle (dataset complet, plus d'epochs, GPU) est proposé comme piste
d'amélioration (Rapport, section 8).

## Résultats (jeu de validation)

| Modèle | Précision | Rappel | F1 | mAP@50 | Latence (ms/image, CPU) | Paramètres (M) | Taille poids |
|---|---|---|---|---|---|---|---|
| **YOLO26-nano** | 0,693 | 0,554 | 0,617* | 0,617 | ~141 | 2,50 | 5,3 Mo |
| **Faster R-CNN** (MobileNetV3-FPN) | 0,255 | 0,231 | 0,243 | 0,311 | ~144 | 18,94 | 76 Mo |
| **SSDlite320** (MobileNetV3) | 0,176 | 0,180 | 0,178 | 0,184 | ~32 | 2,23 | 9,2 Mo |

\* F1 recalculé à partir de precision/recall YOLO (2·P·R/(P+R)).

### Détail par classe (AP@50)

| Classe | YOLO26-nano (mAP50 global) | Faster R-CNN | SSDlite320 |
|---|---|---|---|
| helmet | — (voir mAP50 global) | 0,225 | 0,0001 |
| head | — | 0,709 | 0,553 |
| safety-vest | — | 0,000 | 0,000 |

_Le détail par classe pour YOLO est disponible dans `Modeles/runs/detect/train-4/` (courbes
BoxPR, matrice de confusion)._

## Analyse

- **YOLO26-nano** obtient le meilleur compromis précision/rappel global, cohérent avec un
  entraînement sur le dataset complet et davantage d'epochs — c'est le choix retenu pour
  l'application (section 4.1 du rapport, argument temps réel).
- **Faster R-CNN** dépasse nettement SSDlite sur la classe `head` (AP 0,71 vs 0,55) mais
  reste, comme attendu pour une architecture two-stage, plus lourde (76 Mo, 19M de
  paramètres) sans gain de latence CPU par rapport à YOLO dans ce protocole.
- **SSDlite320** est la plus légère (2,2M de paramètres, 32 ms/image — 4× plus rapide que
  les deux autres) : c'est le candidat le plus adapté à l'edge computing (caméras de
  chantier basse consommation), au prix d'un rappel plus faible.
- **`safety-vest` non détecté (AP≈0) par les deux modèles torchvision** : la classe est
  sous-représentée dans le sous-échantillon de 600 images utilisé pour ces deux modèles
  (déséquilibre déjà documenté section 2.2/3.2 du rapport) — un entraînement sur le
  dataset complet, comme pour YOLO, corrigerait vraisemblablement ce point.

## Reproduire ces résultats

```
python3 comparatif/train_eval.py --model fasterrcnn --epochs 3 --train_size 600 --val_size 150 --cpu
python3 comparatif/train_eval.py --model ssdlite    --epochs 3 --train_size 600 --val_size 150 --cpu
```

Les métriques JSON complètes sont dans `results/fasterrcnn_metrics.json` et
`results/ssdlite_metrics.json`. Les poids entraînés (non versionnés, ~76 Mo et ~9 Mo) sont
régénérés par ces commandes.
