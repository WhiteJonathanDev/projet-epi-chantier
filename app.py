import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-4/weights/best.pt")

model = load_model()

st.title("Détection EPI - Chantier")

confidence = st.slider("Seuil de confiance", 0.0, 1.0, 0.25) 

fichier = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png"])

if fichier is not None:
    img = Image.open(fichier)
    img_array = np.array(img)

    results = model.predict(img, conf=confidence)[0]

    classes_detectees = results.boxes.cls.tolist() if results.boxes is not None else []
    

    conforme = any(c in [0, 2] for c in classes_detectees)

    if conforme:
        st.success("Conforme car EPI détecté")
    else:
        st.error("Non conforme car EPI manquant !")

    img_annotee = results.plot()
    st.image(img_annotee, channels="BGR") 
