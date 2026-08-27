# Guide de révision — Oral BC2 (RNCP40875)

Ce document reprend **chaque indicateur de la grille d'évaluation officielle**
(`sujet/Grilles_Evaluation_BC2_RNCP40875.pdf`) et indique, pour chacun : ce qui est
attendu, où le retrouver dans le rendu, et une phrase-type à dire à l'oral. Objectif :
pouvoir répondre à n'importe quelle question du jury en pointant un élément concret du
projet, pas en improvisant.

**Rappel du barème** : chaque indicateur est noté NA (0) / PA (1) / A (2) / MA (3). Une
compétence est validée si le score ≥ 70 % du max **et** aucun indicateur à NA. Les 9
compétences C3.1 à C5.3 doivent **toutes** être validées pour valider le BC2.

**Périmètre de ce guide** : C3.1 à C4.3 (Projet 1 — traitement de l'image, ce dépôt) +
la dimension orale. **C5.1 à C5.3** (Projet 2 — IA générative) sont couverts par le
projet de ton collègue, à réviser séparément avec lui — ils comptent quand même dans la
validation finale du BC2.

---

## C3.1 — Préparation et nettoyage des données (12 pts)

> Critère : qualité de la préparation des données, outils appropriés, qualité optimale et
> universellement accessible adaptée aux besoins métiers.

| # | Indicateur attendu | Où c'est dans le rendu | Ce que tu peux dire à l'oral |
|---|---|---|---|
| 1 | Outils de transformation/nettoyage adaptés mobilisés efficacement | `Donnees/data_cleaning_report.py` (contrôle intégrité + doublons sur 8 099 images) + `Applications/projet.ipynb` §1.1 (réorganisation SH17→YOLO) et §1.3 (resize, normalisation) | « J'ai écrit un script qui scanne les 8 099 images : 0 corrompue, 0 doublon exact. Le prétraitement (resize 640×640, normalisation) est géré nativement par YOLO, documenté section 2.4 du rapport. » |
| 2 | Données transformées adaptées aux besoins métiers, **en collaboration avec gestionnaires/analystes** | Rapport §9.2 (validation simulée du responsable HSE sur le choix des 3 classes EPI) | « Le choix de restreindre à 3 classes EPI (Axe B) répond à une priorité métier — casque et gilet, les EPI à plus fort impact sécurité — validée dans l'exercice de concertation simulé section 9.2, faute d'accès à un vrai chantier pour ce projet scolaire. » **Point faible assumé : pas de vraie collaboration métier, à dire clairement si le jury pousse la question.** |
| 3 | Étapes de préparation clairement expliquées et documentées | Rapport §2 (Préparation de la donnée), `Donnees/README.md` | « Chaque étape (réorganisation, stats, prétraitement, augmentation) est documentée avec sa justification dans le rapport et reproductible via le notebook. » |
| 4 | Données transformées prêtes pour l'analyse et l'apprentissage automatique | `dataset_yolo_epi/` (6 544 images annotées sur 8 099), effectivement utilisé pour entraîner les 3 modèles comparés §8 | « Le dataset préparé a servi à entraîner 3 modèles différents (YOLO, Faster R-CNN, SSDlite), preuve qu'il est directement exploitable. » |

**Risque principal sur cette compétence** : indicateur 2 (collaboration métier absente,
projet solo). Assume-le plutôt que de bluffer — le jury valorise la lucidité.

---

## C3.2 — Communication infographique et tableaux de bord interactifs (12 pts)

> Critère : cohérence et caractère inclusif de la communication visuelle pour la
> gouvernance.

| # | Indicateur | Où c'est | Ce que tu peux dire |
|---|---|---|---|
| 1 | Visualisations créées en collaboration avec équipes métiers/data analystes, inclusives, adaptées à la gouvernance | `Applications/app.py` (onglet Tableau de bord), palette bleu/orange accessible daltoniens, mode contraste élevé, bilingue FR/EN | « Le dashboard a été conçu avec l'accessibilité en tête : palette validée daltoniens, mode contraste élevé pour écrans extérieurs, bilingue. La collaboration métier reste simulée (même limite qu'en C3.1). » |
| 2 | Graphiques interactifs affichés en temps réel dans le tableau de bord | `app.py` — heatmap zones à risque, timeline des alertes, histogramme classes détectées, tous générés en direct via Plotly à partir de `st.session_state.history` | **Démo à faire en live** : uploader 2-3 images dans l'onglet Détection, montrer les stats se mettre à jour dans Tableau de bord. |
| 3 | Fonctionnalités avancées (filtres, drill-down) exploitées à bon escient | Filtres multiselect par type d'EPI / zone / période (`sel_types`, `sel_zones`, `sel_period` dans `app.py`) | « Les 3 filtres se combinent : on peut isoler les alertes casque de la Zone A sur une période donnée. » |
| 4 | Tableau de bord responsif, facilite la prise de décision | Layout Streamlit `wide`, colonnes adaptatives, métriques en tête de page (taux de conformité, alertes, total) | « Les indicateurs clés (taux de conformité, nombre d'alertes) sont visibles immédiatement en haut de l'onglet, sans scroller. » |

---

## C3.3 — Analyse exploratoire de données (9 pts)

> Critère : pertinence des processus d'EDA pour générer des insights exploitables.

| # | Indicateur | Où c'est | Ce que tu peux dire |
|---|---|---|---|
| 1 | Techniques statistiques et outils adaptés et efficaces | Rapport §2.2/2.3 (ratio classes 118,3, distribution taille bbox), `Donnees/rapport_nettoyage.json` (distribution réelle par classe sur le sous-set EPI) | « Deux angles d'analyse : déséquilibre de classes (ratio 118,3 sur 17 classes) et taille des objets (majorité de petites bbox), qui ont chacun influencé une décision technique différente (Axe B et paramètre `scale` YOLO). » |
| 2 | Insights pertinents pour les besoins métiers, en collaboration avec équipes métiers | Rapport §3.2 (Insights métiers) : classes rares sous-représentées, petits objets = EPI détectés à distance plus difficiles | « L'insight le plus actionnable : les EPI détectés à distance (caméra large chantier) seront moins bien captés — ça doit orienter le positionnement des caméras en déploiement réel. » |
| 3 | Étapes d'analyse documentées et alignées sur les objectifs stratégiques | Rapport §3, notebook §1.2 | « Chaque insight de l'EDA a débouché sur une décision documentée : restriction à 3 classes, choix des augmentations. Rien n'est fait "pour la forme". » |

---

## C4.1 — Définition de la stratégie d'intégration de l'IA (9 pts)

> Critère : cohérence et faisabilité de la stratégie au regard des cas d'usage et de leur
> impact métier.

| # | Indicateur | Où c'est | Ce que tu peux dire |
|---|---|---|---|
| 1 | Cas d'usage pertinents et alignés, **en concertation avec responsables IA/métiers** | Rapport §9.1 (cas d'usage) + §9.2 (validation simulée à 3 profils : chef de chantier, HSE, IT) | « J'ai simulé une consultation à 3 rôles métier différents pour prioriser casque + gilet plutôt que les 17 classes — c'est documenté comme un exercice assumé, pas une vraie validation terrain. » |
| 2 | Impact des cas d'usage clairement évalué et expliqué | Rapport §9.3 (impact métier) + tableau SWOT | « L'impact est structuré en SWOT : le bénéfice principal est le passage d'un contrôle ponctuel à une supervision continue journalisée ; le risque principal est l'acceptabilité (perception de surveillance). » |
| 3 | Stratégie matérialisée par une feuille de route réalisable, répondant aux objectifs de transformation | Rapport §9.4 (feuille de route 3 phases avec ressources et indicateurs) + §9.5 (ressources GPU) | « La feuille de route est en 3 phases (test 1-2 mois, généralisation 3-6 mois, optimisation continue), chacune avec ses ressources humaines/matérielles et son indicateur de succès — pas juste une intention. » |

---

## C4.2 — Développement de modèles prédictifs (12 pts)

> Critère : efficacité et cohérence du modèle prédictif pour fournir des insights
> exploitables.

| # | Indicateur | Où c'est | Ce que tu peux dire |
|---|---|---|---|
| 1 | Données bien prétraitées et adaptées à la modélisation prédictive | Rapport §2.4-2.5, `dataset_yolo_epi/` | (cf. C3.1) |
| 2 | Algorithmes ML testés et choix justifié | Rapport §4.1 (choix YOLO justifié : temps réel, dataset déséquilibré, transfer learning) + §8 (2 alternatives testées) | « YOLO a été choisi pour le temps réel, mais j'ai testé 2 architectures alternatives pour objectiver ce choix plutôt que de le poser a priori — voir C4.3. » |
| 3 | Codes implémentés fonctionnels et sans erreur | `Applications/projet.ipynb` (exécuté, sorties visibles), `Applications/app.py` (testé, démarre sans erreur — capture faite le 2026-08-10) | **Montrer le notebook exécuté avec les sorties, ou relancer l'app en live si possible.** |
| 4 | Résultats du modèle répondent aux objectifs du cas métier | Rapport §5-6 (mAP@50 0,616 sur EPI 3 classes, règle de conformité en inférence, bandeau d'alerte) | « Le modèle EPI atteint 0,616 de mAP@50 avec un rappel priorisé (métrique la plus importante ici, cf. §5), et déclenche une vraie alerte visuelle testée sur image et vidéo. » |

---

## C4.3 — Évaluation comparative des performances des modèles (9 pts)

> Critère : exhaustivité et cohérence de la comparaison, en intégrant l'éco-responsabilité.

| # | Indicateur | Où c'est | Ce que tu peux dire |
|---|---|---|---|
| 1 | Plusieurs modèles développés et comparés avec métriques appropriées | Rapport §8 + `Modeles/comparatif/RESULTATS.md` : YOLO26-nano, Faster R-CNN (MobileNetV3-FPN), SSDlite320 — précision/rappel/F1/mAP@50/latence/paramètres, même protocole (dataset complet, 30 epochs, GPU L4) | « 3 architectures réellement entraînées et évaluées avec le même protocole, pas juste citées : YOLO (mAP 0,616), Faster R-CNN (0,561), SSDlite (0,370). » |
| 2 | Améliorations proposées pertinentes (ex. gestion du sur-apprentissage) | Rapport §8, paragraphe « Gestion du sur-apprentissage » (early stopping `patience=10` pour YOLO) | « Le early stopping évite le sur-apprentissage sur YOLO. Pour les modèles de comparaison, la première itération sur 600 images avait un vrai problème d'échantillon (pas assez d'exemples de `safety-vest`) — corrigé en ré-entraînant sur le dataset complet via un serveur GPU. » |
| 3 | Modèle final validé par les parties prenantes, éco-responsabilité prise en compte | Rapport §8 paragraphe « Validation du modèle retenu » (lien avec §9.2) + tableau paramètres (proxy énergie : YOLO 2,5M vs Faster R-CNN 18,9M de paramètres) | « Le nombre de paramètres sert de proxy d'empreinte énergétique — YOLO et SSDlite sont 8× plus légers que Faster R-CNN. Le choix final de YOLO est cohérent avec les critères exprimés par le "responsable IT" simulé (pas de dépendance matérielle lourde). » |

**C'est la compétence qui a le plus progressé récemment.** Sois prêt à raconter
l'évolution : une première version du comparatif (600 images, 3 epochs, CPU) donnait un
AP à 0,000 sur `safety-vest` pour Faster R-CNN et SSDlite — pas un défaut d'architecture,
mais un échantillon trop réduit pour cette classe rare. Après accès à un serveur GPU, le
ré-entraînement sur le dataset complet (30 epochs, même protocole pour les 3 modèles) a
fait passer `safety-vest` à 0,309 (Faster R-CNN) et 0,185 (SSDlite). **Raconte cette
itération** — ça montre une vraie démarche scientifique (diagnostic → hypothèse →
correction → nouvelle mesure), pas juste un résultat final.

---

## Oral — Présentation et défense du rapport (12 pts)

| # | Indicateur | Comment t'y préparer |
|---|---|---|
| 1 | Présentation structurée, claire, synthétique sur C3.1 à C5.3, dans le temps imparti (45 min max) | Prépare un plan de présentation qui suit l'ordre du rapport (données → modélisation → comparatif → stratégie), chronomètre-toi à l'avance. |
| 2 | Argumentation rigoureuse des choix méthodologiques et techniques au regard des besoins métiers | Pour chaque choix (YOLO, 3 classes EPI, seuil de confiance, architecture du dashboard), sache dire *pourquoi ce choix et pas un autre* — les tableaux ci-dessus te donnent la réponse pour chaque point. |
| 3 | Rapport (et support) structuré, clair, lisible, universellement accessible | `Documentations/Rapport.pdf` — vérifie qu'il s'ouvre bien et que les images sont lisibles avant le jour J. |
| 4 | Défendre ses choix avec assurance, répondre avec rigueur, s'adapter au profil du jury | Anticipe les questions pièges : *« Le comparatif est-il fiable ? »* (oui, désormais même protocole GPU pour les 3 modèles — raconte l'itération 600→4708 images), *« Le dashboard a-t-il été testé par de vrais utilisateurs ? »* (non, validation simulée, assumé §9.2). |

---

## Questions pièges à anticiper (et où trouver la réponse)

| Question probable du jury | Réponse courte | Détail |
|---|---|---|
| Pourquoi avoir réentraîné alors que le dataset est déjà annoté ? | COCO (poids de départ) ne connaît pas les classes EPI — le réentraînement adapte la tête de détection à ces nouvelles classes, ce n'est pas une répétition inutile | Section "pourquoi nettoyer/réentraîner" de la conversation, à reformuler dans tes mots |
| Le dataset est-il vraiment "nettoyé" si vous n'avez rien supprimé ? | Le nettoyage = contrôle qualité (0 image corrompue, 0 doublon sur 8 099), pas suppression arbitraire — un dataset propre au départ n'a pas besoin d'être charcuté | `Donnees/rapport_nettoyage.json` |
| Le comparatif de modèles était-il fiable avec seulement 600 images ? | Non initialement (limite assumée), corrigé depuis : ré-entraîné sur GPU avec le dataset complet et un protocole identique pour les 3 modèles | Rapport §8 + §9.5, `Modeles/comparatif/RESULTATS.md` |
| Le tableau de bord a-t-il été validé par de vrais chefs de chantier ? | Non — validation simulée avec des profils types, assumée comme telle, à refaire en phase 1 de la feuille de route | Rapport §9.2 |
| Qu'est-ce qui prouve que l'app fonctionne réellement ? | Démo live, ou capture de l'exécution réussie du 2026-08-10 (démarrage sans erreur, HTTP 200) | `Applications/app.py` |

---

## Check-list avant l'oral

- [ ] Relire `Documentations/Rapport.pdf` en entier une fois à voix haute
- [ ] Savoir citer de mémoire : mAP@50 du modèle EPI (0,616), les 3 modèles comparés et
      leur mAP@50 respectif (YOLO 0,616 / Faster R-CNN 0,561 / SSDlite 0,370)
- [ ] Pouvoir lancer `streamlit run Applications/app.py` en live si le jury demande une démo
- [ ] Avoir un lien de déploiement Streamlit Cloud fonctionnel (à faire — voir README)
- [ ] Savoir expliquer en 1 phrase chaque limite assumée (échantillon réduit, validation
      simulée, collaboration métier simulée) — les citer avant qu'on te les reproche
- [ ] Réviser séparément le Projet 2 (C5.1-C5.3) avec ton collègue
