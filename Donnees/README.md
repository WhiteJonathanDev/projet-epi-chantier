# Données — dataset SH17 / EPI

## Origine

[SH17 (Safe Human 17)](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection) :
8 099 images annotées, 75 994 instances, 17 classes liées aux EPI et à la sécurité humaine
sur des environnements industriels et des chantiers de construction.

Le dataset brut n'est pas inclus dans ce dépôt (~13 Go). Pour le reconstruire :

1. Télécharger le dataset SH17 depuis le lien ci-dessus (ou depuis
   [GitHub](https://github.com/ahmadmughees/SH17dataset)).
2. Placer les données brutes dans un dossier `dataset_kaggle/` à la racine de `Applications/`.
3. Exécuter `Applications/projet.ipynb` depuis le début (section 1) : il reconstruit
   automatiquement `dataset_yolo/` (17 classes) et `dataset_yolo_epi/` (3 classes EPI —
   helmet, head, safety-vest, cf. Axe B) au format YOLO
   (`images/{train,val,test}/`, `labels/{train,val,test}/`).

## Nettoyage et transformation (détail méthodologique)

Voir `Applications/projet.ipynb`, section 1, et `Documentations/Rapport.md`/`.pdf`, section 1,
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

## Contenu de ce dossier

Les images brutes (13 Go) ne sont pas versionnées, mais tout le reste du travail de
nettoyage/annotation l'est :

- **`annotations_epi/labels/{train,val,test}/`** — les 6 544 fichiers d'annotation YOLO
  (3 classes EPI : helmet, head, safety-vest) réellement utilisés pour entraîner les
  modèles. Chaque fichier `.txt` correspond à une image (`classe xc yc largeur hauteur`,
  coordonnées normalisées) — c'est le livrable « annotations » à proprement parler,
  indépendant des images elles-mêmes.
- **`echantillon_annote/`** — 12 images du dataset avec les bounding boxes dessinées
  dessus, pour illustrer visuellement la qualité des annotations (pas un échantillon de
  travail, juste un aperçu).
- **`rapport_nettoyage.json`** (généré par `data_cleaning_report.py`, exécuté sur les
  8 099 images) — résultat réel du contrôle qualité automatique :

  | Contrôle | Résultat |
  |---|---|
  | Images scannées | 8 099 |
  | Images corrompues/illisibles | 0 |
  | Doublons exacts (hash de contenu) | 0 |
  | Images avec au moins une annotation EPI | 6 544 (train 4 708, val 1 327, test 509) |
  | Images sans annotation EPI (aucun helmet/head/safety-vest visible) | 1 555 |
  | Instances par classe (train+val+test) | head : 11 985 · helmet : 927 · safety-vest : 530 |

  Ce dernier chiffre confirme, sur le sous-ensemble EPI, le déséquilibre déjà identifié
  section 2.2 du rapport (classe `head` trois fois plus fréquente à elle seule que
  `helmet`+`safety-vest` réunies) — c'est ce déséquilibre qui explique la faible
  performance sur `safety-vest` du comparatif multi-modèles (`Modeles/comparatif/RESULTATS.md`).
  Reproductible avec `python3 Donnees/data_cleaning_report.py`.
- **`annotation_quality_report.json`** (généré par `annotation_quality.py`, exécuté sur
  les 6 544 fichiers d'annotation) — contrôle de la qualité des **boîtes englobantes**
  elles-mêmes (pas seulement des images), avec correction automatique :

  | Contrôle | Résultat |
  |---|---|
  | Boîtes analysées | 13 442 |
  | Boîtes avec coordonnées hors [0,1] (bbox débordant de l'image) | 187 → **corrigées** (clip à [0,1]) |
  | Boîtes dégénérées (largeur ou hauteur quasi nulle) | 17 → **supprimées** |
  | Boîtes à ratio d'aspect extrême (>15:1) | 0 signalée |
  | Fichiers modifiés par la correction | 187 sur 6 544 (2,9 %) |

  Les annotations corrigées sont dans `annotations_epi_corrigees/` (dossier séparé, les
  originaux ne sont jamais écrasés). Les modèles actuels ont été entraînés sur les
  annotations d'origine : l'impact des 187 boîtes corrigées (1,4 % du total) est jugé
  marginal au vu de leur faible proportion, mais un ré-entraînement sur
  `annotations_epi_corrigees/` reste la suite logique avant un déploiement en production.
  Reproductible avec `python3 Donnees/annotation_quality.py`.
