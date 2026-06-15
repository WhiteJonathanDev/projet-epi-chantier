# Détection automatique d'EPI sur chantier

Projet de traitement d'image : détection des Équipements de Protection Individuelle (EPI) sur des images et vidéos de chantier, avec déclenchement d'une alerte visuelle en cas de non-conformité.

## Structure du projet

-  projet.ipynb   # Notebook principal (pipeline complet, déjà exécuté avec tous les résultats)
- app.py  # Interface Streamlit (Axe E)
- Rapport.pdf   # Rapport écrit
- sh17.yaml  # Config dataset complet (17 classes)
- sh17_epi.yaml # Config dataset EPI (3 classes - Axe B)
- runs/detect/ # Poids entraînés et résultats d'entraînement/validation YOLO
- requirements.txt

## Dataset


Pour reproduire l'entraînement à partir de zéro :
1. Télécharger le dataset SH17 : https://github.com/ahmadmughees/SH17dataset (ou via Kaggle)
2. Placer les données brutes dans un dossier `dataset_kaggle/` à la racine du projet
3. Exécuter le notebook depuis le début (section 1) : il reconstruit `dataset_yolo/` et `dataset_yolo_epi/` automatiquement

Les poids déjà entraînés (`runs/detect/train/weights/best.pt` pour le modèle 17 classes, `runs/detect/train-4/weights/best.pt` pour le modèle EPI) sont inclus, donc **l'interface Streamlit (`app.py`) fonctionne directement sans dataset**.

## Reproductibilité

Les entraînements utilisent `seed=0` et `deterministic=True` (valeurs par défaut d'Ultralytics, visibles dans `runs/detect/*/args.yaml`), garantissant des résultats reproductibles.

## Installation

pip install -r requirements.txt


## Utilisation

### Notebook

Ouvrir `projet.ipynb` et exécuter les cellules dans l'ordre. Le notebook couvre l'intégralité du pipeline :

1. Préparation et exploration du dataset SH17
2. Pré-traitement et augmentation des données
3. Entraînement du modèle YOLO
4. Évaluation (mAP, précision, rappel, F1-score par classe)
5. Inférence sur image et vidéo avec alerte de non-conformité (bandeau rouge)
6. Partie 2 : Axe B (entraînement restreint aux 3 classes EPI : helmet, head, safety-vest)

### Interface Streamlit (Axe E)

streamlit run app.py

Permet de charger une image, de régler le seuil de confiance, et d'afficher la détection annotée ainsi qu'un indicateur de conformité global.
