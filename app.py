# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from utils.audio import audio_to_mel
from utils.preprocess import preprocess_mel

# ===============================
# Cargar modelo entrenado
# ===============================
MODEL_PATH = "model/accent_mobilenetv2.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Clases de salida
CLASSES = ['andino', 'mexico', 'espana']

# ===============================
# Interfaz Streamlit
# ===============================
st.title("🎤 Detector de Acento Español")
st.write("Sube un archivo de audio y el modelo predirá el acento.")

# Subida de archivo
uploaded_file = st.file_uploader("Selecciona un archivo .wav", type=["wav"])

if uploaded_file is not None:
    # Guardar temporalmente el archivo
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio("temp.wav", format="audio/wav")
    
    # Convertir audio a mel y preprocesar
    mel = audio_to_mel("temp.wav")
    mel = preprocess_mel(mel)
    mel = np.expand_dims(mel, axis=0)  # Añadir batch dimension
    
    # Predicción
    pred = model.predict(mel)
    pred_class = CLASSES[np.argmax(pred)]
    confidence = np.max(pred)
    
    st.success(f"Predicción: **{pred_class}** (Confianza: {confidence:.2f})")
