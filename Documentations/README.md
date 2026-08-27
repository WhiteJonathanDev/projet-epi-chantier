# Détection automatique d'EPI sur chantier

Projet de traitement d'image : détection des Équipements de Protection Individuelle (EPI)
sur des images et vidéos de chantier, avec déclenchement d'une alerte visuelle en cas de
non-conformité.

Bloc 2 – Projet 1 (RNCP40875 – Expert en ingénierie de données).

## Structure du rendu (§6.3 du sujet)

```
rendu/
├── Donnees/            # Description, nettoyage et reproduction du dataset SH17 / EPI
├── Modeles/             # Configs YOLO, poids entraînés, codes et résultats du comparatif
├── Applications/        # App Streamlit, notebook d'analyse/entraînement, dépendances
└── Documentations/       # README (ce fichier) et rapport écrit
```

## Dataset

Le dataset brut (SH17, ~13 Go) n'est pas inclus dans ce dépôt pour des raisons de taille.
Voir [`Donnees/README.md`](../Donnees/README.md) pour la description du nettoyage/préparation
et les instructions de reproduction.

Les poids déjà entraînés sont inclus dans `Modeles/runs/detect/`
(`train/weights/best.pt` : modèle 17 classes, `train-4/weights/best.pt` : modèle EPI 3 classes),
donc **l'application Streamlit fonctionne directement sans télécharger le dataset**.

## Reproductibilité

Les entraînements YOLO utilisent `seed=0` et `deterministic=True` (valeurs par défaut
d'Ultralytics, visibles dans `Modeles/runs/detect/*/args.yaml`).

## Installation

```
pip install -r Applications/requirements.txt
```

## Utilisation

### Notebook (préparation, entraînement, évaluation YOLO)

Ouvrir `Applications/projet.ipynb` et exécuter les cellules dans l'ordre (le dataset doit être
reconstruit au préalable, voir `Donnees/README.md`). Le notebook couvre :

1. Préparation et exploration du dataset SH17
2. Pré-traitement et augmentation des données
3. Entraînement du modèle YOLO (17 classes)
4. Évaluation (mAP, précision, rappel, F1-score par classe)
5. Inférence sur image et vidéo avec alerte de non-conformité
6. Axe B : entraînement restreint aux 3 classes EPI (helmet, head, safety-vest)

### Comparatif multi-modèles (YOLO / Faster R-CNN / MobileNet-SSD)

Scripts et résultats dans `Modeles/comparatif/` — voir `Modeles/comparatif/RESULTATS.md`
pour le tableau comparatif (précision, rappel, F1, mAP@50, latence, taille du modèle).

### Application Streamlit (dashboard)

```
streamlit run Applications/app.py
```

Fonctionnalités : analyse d'image ou de vidéo, choix du modèle et du seuil de confiance,
alertes de non-conformité, tableau de bord (taux de conformité, heatmap des zones à
risque, timeline des alertes) avec filtres par type d'EPI / zone / période, guide
d'utilisation intégré, mode contraste élevé et bilingue FR/EN pour l'accessibilité.

**Lien de déploiement** : _à compléter après déploiement sur Streamlit Community Cloud_
(voir `Documentations/Rapport.md`, section Livrables, pour la marche à suivre).

## Tests unitaires

```
pip install -r Applications/requirements-dev.txt
pytest Applications/tests/                  # logique de conformité EPI (9 tests)

pip install -r Modeles/comparatif/requirements.txt
pytest Modeles/comparatif/tests/            # métriques + dataset du comparatif (13 tests)
```
