# Données — dataset SH17 / EPI

## Origine

[SH17 (Safe Human 17)](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection) :
8 099 images annotées, 75 994 instances, 17 classes liées aux EPI et à la sécurité humaine
sur des environnements industriels et des chantiers de construction.

Le dataset brut n'est pas inclus dans ce dépôt (~13 Go). Pour le reconstruire :

1. Télécharger le dataset SH17 depuis le lien ci-dessus (ou depuis
   [GitHub](https://github.com/ahmadmughees/SH17dataset)).
2. Placer les données brutes dans un dossier `dataset_kaggle/` à la racine de `Modeles/`.
3. Exécuter `Modeles/projet.ipynb` depuis le début (section 1) : il reconstruit
   automatiquement `dataset_yolo/` (17 classes) et `dataset_yolo_epi/` (3 classes EPI —
   helmet, head, safety-vest, cf. Axe B) au format YOLO
   (`images/{train,val,test}/`, `labels/{train,val,test}/`).

## Nettoyage et transformation (détail méthodologique)

Voir `Modeles/projet.ipynb`, section 1, et `Documentation/Rapport.md`/`.pdf`, section 1,
pour le détail complet. Résumé :

- **Réorganisation** : passage du format SH17 (dossier unique + listes `train.txt`/`val.txt`)
  au format YOLO standard (`images/`, `labels/` par split).
- **Exploration/statistiques** : distribution des instances par classe (déséquilibre
  fort : ratio classe majoritaire/minoritaire de 118,3), distribution de la taille des
  bounding boxes (majorité de petits objets).
- **Prétraitement** : redimensionnement à 640×640 (résolution d'entrée YOLO), normalisation
  des pixels ([0,255] → [0,1]), pas de débruitage nécessaire (aucun artefact constaté).
- **Augmentation** : variations d'éclairage (nuit, éblouissement, contre-jour), changements
  d'angle de vue (rotation, flip), occlusions partielles simulées — puis activées à
  l'entraînement via les paramètres YOLO `fliplr=0.5` et `hsv_v=0.5`.
- **Choix méthodologiques justifiés** : aucune image supprimée manuellement (pas d'images
  floues/mal cadrées identifiées lors de l'exploration visuelle) ; le déséquilibre de
  classes est traité en aval par le choix d'un sous-ensemble ciblé de 3 classes (Axe B)
  plutôt que par sur-échantillonnage, pour rester représentatif des besoins métier
  prioritaires (casque, tête nue, gilet).

## Format d'export

`dataset_yolo/` et `dataset_yolo_epi/` sont au format YOLO (compatible Ultralytics et,
après conversion — voir `Modeles/comparatif/dataset_torchvision.py` —, avec les modèles
`torchvision.models.detection` utilisés dans le comparatif multi-modèles).
