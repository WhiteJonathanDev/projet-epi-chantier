"""Tableau de bord de detection EPI - Interface Streamlit (Axe E).

Fonctionnalites : upload image/video, filtres (type EPI / zone / periode),
statistiques de conformite, heatmap des zones a risque, timeline des alertes,
guide d'utilisation integre, mode contraste eleve (accessibilite daltoniens),
bilingue FR/EN.
"""
from collections import Counter
from pathlib import Path

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR.parent / "Modeles" / "runs" / "detect"

# --- Palette accessible (daltoniens) : bleu/orange plutot que rouge/vert ---
COLOR_OK = "#0072B2"      # bleu
COLOR_ALERT = "#E69F00"   # orange
COLOR_ALERT_HC = "#D55E00"  # orange fonce (contraste eleve)

TEXT = {
    "fr": {
        "title": "Détection EPI - Chantier",
        "subtitle": "Contrôle automatique du port des Équipements de Protection Individuelle",
        "tab_detect": "Détection",
        "tab_dashboard": "Tableau de bord",
        "tab_guide": "Guide d'utilisation",
        "sidebar_settings": "Réglages",
        "confidence": "Seuil de confiance",
        "zone": "Zone du chantier",
        "model_choice": "Modèle de détection",
        "high_contrast": "Mode contraste élevé",
        "language": "Langue / Language",
        "upload_image": "Charger une image",
        "upload_video": "Charger une vidéo",
        "mode": "Type de média",
        "conforme": "Conforme : EPI détecté",
        "non_conforme": "Non conforme : EPI manquant !",
        "missing": "Équipements manquants détectés",
        "no_detection": "Aucune détection sur ce média",
        "history_empty": "Aucune donnée pour l'instant. Analysez une image ou une vidéo dans l'onglet Détection.",
        "filter_type": "Filtrer par type d'EPI",
        "filter_zone": "Filtrer par zone",
        "filter_period": "Filtrer par période",
        "stat_total": "Analyses totales",
        "stat_conformity": "Taux de conformité",
        "stat_alerts": "Alertes non-conformité",
        "heatmap_title": "Heatmap des zones à risque (alertes par zone)",
        "timeline_title": "Timeline des alertes",
        "class_dist_title": "Répartition des classes détectées",
        "clear_history": "Réinitialiser l'historique",
        "guide_text": """
### Comment utiliser ce tableau de bord

1. **Onglet Détection** : choisissez une image ou une vidéo de chantier, réglez le seuil
   de confiance et la zone du chantier concernée, puis lancez l'analyse.
2. Le modèle annote l'image/la vidéo et affiche un bandeau de conformité
   (bleu = conforme, orange = non conforme) ainsi que la liste des EPI manquants.
3. **Onglet Tableau de bord** : consultez les statistiques cumulées de la session
   (taux de conformité, heatmap des zones à risque, timeline des alertes), filtrables
   par type d'EPI, par zone et par période.
4. **Accessibilité** : la palette bleu/orange reste distinguable pour les daltoniens ;
   le mode contraste élevé accentue les bandeaux d'alerte pour les écrans extérieurs.
5. Les statistiques sont propres à la session du navigateur ; utilisez
   « Réinitialiser l'historique » pour repartir de zéro.
        """,
    },
    "en": {
        "title": "PPE Detection - Construction Site",
        "subtitle": "Automatic Personal Protective Equipment compliance monitoring",
        "tab_detect": "Detection",
        "tab_dashboard": "Dashboard",
        "tab_guide": "User Guide",
        "sidebar_settings": "Settings",
        "confidence": "Confidence threshold",
        "zone": "Site zone",
        "model_choice": "Detection model",
        "high_contrast": "High-contrast mode",
        "language": "Langue / Language",
        "upload_image": "Upload an image",
        "upload_video": "Upload a video",
        "mode": "Media type",
        "conforme": "Compliant: PPE detected",
        "non_conforme": "Non-compliant: missing PPE!",
        "missing": "Missing equipment detected",
        "no_detection": "No detection on this media",
        "history_empty": "No data yet. Analyze an image or video in the Detection tab.",
        "filter_type": "Filter by PPE type",
        "filter_zone": "Filter by zone",
        "filter_period": "Filter by period",
        "stat_total": "Total analyses",
        "stat_conformity": "Compliance rate",
        "stat_alerts": "Non-compliance alerts",
        "heatmap_title": "Risk zone heatmap (alerts per zone)",
        "timeline_title": "Alert timeline",
        "class_dist_title": "Detected class distribution",
        "clear_history": "Reset history",
        "guide_text": """
### How to use this dashboard

1. **Detection tab**: pick a site image or video, set the confidence threshold and
   the relevant site zone, then run the analysis.
2. The model annotates the image/video and shows a compliance banner
   (blue = compliant, orange = non-compliant) plus the list of missing PPE.
3. **Dashboard tab**: review cumulative session statistics (compliance rate, risk
   zone heatmap, alert timeline), filterable by PPE type, zone and period.
4. **Accessibility**: the blue/orange palette stays distinguishable for colorblind
   users; high-contrast mode boosts alert banners for outdoor screens.
5. Statistics are scoped to your browser session; use "Reset history" to start over.
        """,
    },
}

CLASS_NAMES_EPI = {0: "helmet", 1: "head", 2: "safety-vest"}
COMPLIANT_CLASSES_EPI = [0, 2]  # helmet, safety-vest presents = conforme
MISSING_HINT_EPI = {1: "helmet"}  # "head" detecte sans casque -> casque manquant

AVAILABLE_MODELS = {
    "YOLO - EPI (3 classes)": MODELS_DIR / "train-4" / "weights" / "best.pt",
    "YOLO - SH17 complet (17 classes)": MODELS_DIR / "train" / "weights" / "best.pt",
}


@st.cache_resource
def load_model(path_str):
    return YOLO(path_str)


def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dict rows


def run_detection(model, media, confidence):
    results = model.predict(media, conf=confidence, verbose=False)[0]
    classes = results.boxes.cls.tolist() if results.boxes is not None else []
    return results, classes


def record_detection(zone, classes, conforme, timestamp):
    st.session_state.history.append({
        "timestamp": timestamp,
        "zone": zone,
        "conforme": conforme,
        "n_detections": len(classes),
        "classes": [CLASS_NAMES_EPI.get(int(c), str(int(c))) for c in classes],
    })


def main():
    st.set_page_config(page_title="EPI Chantier", page_icon="🦺", layout="wide")
    init_state()

    lang = st.sidebar.selectbox("🌐 " + TEXT["fr"]["language"], ["fr", "en"],
                                 format_func=lambda x: "Français" if x == "fr" else "English")
    T = TEXT[lang]

    high_contrast = st.sidebar.checkbox("🔆 " + T["high_contrast"], value=False)
    alert_color = COLOR_ALERT_HC if high_contrast else COLOR_ALERT

    st.sidebar.header(T["sidebar_settings"])
    model_label = st.sidebar.selectbox(T["model_choice"], list(AVAILABLE_MODELS.keys()))
    model_path = AVAILABLE_MODELS[model_label]
    confidence = st.sidebar.slider(T["confidence"], 0.0, 1.0, 0.25)
    zone = st.sidebar.text_input(T["zone"], value="Zone A")

    if not model_path.exists():
        st.sidebar.error(f"Poids introuvables : {model_path}")
        st.stop()
    model = load_model(str(model_path))
    is_epi_model = "EPI" in model_label

    st.title("🦺 " + T["title"])
    st.caption(T["subtitle"])

    tab_detect, tab_dashboard, tab_guide = st.tabs([T["tab_detect"], T["tab_dashboard"], T["tab_guide"]])

    # ---------------- Onglet Detection ----------------
    with tab_detect:
        media_type = st.radio(T["mode"], ["image", "video"], horizontal=True,
                               format_func=lambda x: T["upload_image"] if x == "image" else T["upload_video"])

        if media_type == "image":
            fichier = st.file_uploader(T["upload_image"], type=["jpg", "jpeg", "png"])
            if fichier is not None:
                img = Image.open(fichier)
                results, classes = run_detection(model, img, confidence)

                conforme = True
                missing = []
                if is_epi_model and classes:
                    conforme = any(c in COMPLIANT_CLASSES_EPI for c in classes)
                    if not conforme:
                        missing = [MISSING_HINT_EPI.get(int(c)) for c in classes if int(c) in MISSING_HINT_EPI]

                col1, col2 = st.columns([2, 1])
                with col1:
                    img_annotee = results.plot()
                    st.image(img_annotee, channels="BGR", use_container_width=True)
                with col2:
                    if not classes:
                        st.info(T["no_detection"])
                    elif conforme:
                        st.markdown(f"<div style='background-color:{COLOR_OK};color:white;padding:12px;"
                                    f"border-radius:8px;font-weight:bold;'>✅ {T['conforme']}</div>",
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background-color:{alert_color};color:white;padding:12px;"
                                    f"border-radius:8px;font-weight:bold;'>⚠️ {T['non_conforme']}</div>",
                                    unsafe_allow_html=True)
                        if missing:
                            st.write(f"**{T['missing']}** : {', '.join(sorted(set(missing)))}")

                    st.write(f"Zone : **{zone}** — Modèle : **{model_label}**")

                record_detection(zone, classes, conforme, pd.Timestamp.now())

        else:
            fichier = st.file_uploader(T["upload_video"], type=["mp4", "mov", "avi"])
            sample_rate = st.slider("Échantillonnage (1 frame analysée / N)", 1, 30, 10)
            if fichier is not None:
                tmp_path = APP_DIR / "_tmp_upload.mp4"
                with open(tmp_path, "wb") as f:
                    f.write(fichier.read())

                cap = cv2.VideoCapture(str(tmp_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0)
                frame_placeholder = st.empty()
                status_placeholder = st.empty()

                frame_idx, alerts_in_video, analysed = 0, 0, 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_rate == 0:
                        results, classes = run_detection(model, frame, confidence)
                        conforme = True
                        if is_epi_model and classes:
                            conforme = any(c in COMPLIANT_CLASSES_EPI for c in classes)
                        if not conforme:
                            alerts_in_video += 1
                        analysed += 1
                        annotated = results.plot()
                        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)
                        status = T["conforme"] if conforme else T["non_conforme"]
                        color = COLOR_OK if conforme else alert_color
                        status_placeholder.markdown(
                            f"<div style='background-color:{color};color:white;padding:8px;"
                            f"border-radius:6px;'>{status} (frame {frame_idx})</div>",
                            unsafe_allow_html=True)
                        record_detection(zone, classes, conforme, pd.Timestamp.now())
                    frame_idx += 1
                    if total_frames:
                        progress.progress(min(frame_idx / total_frames, 1.0))
                cap.release()
                tmp_path.unlink(missing_ok=True)
                st.success(f"Vidéo analysée : {analysed} frames échantillonnées, {alerts_in_video} alertes détectées.")

    # ---------------- Onglet Tableau de bord ----------------
    with tab_dashboard:
        history = st.session_state.history
        if not history:
            st.info(T["history_empty"])
        else:
            df = pd.DataFrame(history)
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            all_zones = sorted(df["zone"].unique().tolist())
            all_classes = sorted({c for row in df["classes"] for c in row})

            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                sel_types = st.multiselect(T["filter_type"], all_classes, default=all_classes)
            with fc2:
                sel_zones = st.multiselect(T["filter_zone"], all_zones, default=all_zones)
            with fc3:
                date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()
                sel_period = st.date_input(T["filter_period"], value=(date_min, date_max))

            mask = df["zone"].isin(sel_zones) & df["classes"].apply(lambda cs: (not cs) or any(c in sel_types for c in cs))
            if isinstance(sel_period, tuple) and len(sel_period) == 2:
                mask &= (df["timestamp"].dt.date >= sel_period[0]) & (df["timestamp"].dt.date <= sel_period[1])
            fdf = df[mask]

            m1, m2, m3 = st.columns(3)
            m1.metric(T["stat_total"], len(fdf))
            conformity_rate = (fdf["conforme"].mean() * 100) if len(fdf) else 0
            m2.metric(T["stat_conformity"], f"{conformity_rate:.0f}%")
            m3.metric(T["stat_alerts"], int((~fdf["conforme"]).sum()))

            colA, colB = st.columns(2)
            with colA:
                st.subheader(T["timeline_title"])
                timeline = fdf.set_index("timestamp").resample("1min")["conforme"].agg(["count", "sum"])
                timeline["alerts"] = timeline["count"] - timeline["sum"]
                if len(timeline):
                    fig = px.bar(timeline.reset_index(), x="timestamp", y="alerts",
                                 color_discrete_sequence=[alert_color])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption(T["history_empty"])

            with colB:
                st.subheader(T["heatmap_title"])
                zone_counts = fdf[~fdf["conforme"]]["zone"].value_counts().reset_index()
                zone_counts.columns = ["zone", "alertes"]
                if len(zone_counts):
                    fig2 = px.bar(zone_counts, x="zone", y="alertes",
                                  color="alertes", color_continuous_scale="Oranges")
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.caption(T["history_empty"])

            st.subheader(T["class_dist_title"])
            class_counter = Counter(c for row in fdf["classes"] for c in row if c in sel_types)
            if class_counter:
                cdf = pd.DataFrame(class_counter.items(), columns=["classe", "occurrences"])
                fig3 = px.bar(cdf, x="classe", y="occurrences", color_discrete_sequence=[COLOR_OK])
                st.plotly_chart(fig3, use_container_width=True)

            if st.button(T["clear_history"]):
                st.session_state.history = []
                st.rerun()

    # ---------------- Onglet Guide ----------------
    with tab_guide:
        st.markdown(T["guide_text"])


if __name__ == "__main__":
    main()
