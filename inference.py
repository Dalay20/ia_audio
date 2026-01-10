# utils/inference.py
import numpy as np
import tensorflow as tf
from utils.audio import audio_to_mel
from utils.preprocess import preprocess_mel

# Cargar el modelo entrenado (asegúrate que la ruta sea correcta)
MODEL_PATH = "model/accent_mobilenetv2.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Lista de clases
CLASSES = ['andino', 'mexico', 'espana']

def predict_accent(audio_path):
    """
    Predice el acento de un archivo de audio.

    Args:
        audio_path (str): Ruta al archivo .wav

    Returns:
        str: Acento predicho ('andino', 'mexico' o 'espana')
    """
    # 1. Convertir audio a Mel spectrogram
    mel = audio_to_mel(audio_path)

    # 2. Preprocesar el Mel para MobileNetV2
    mel_preprocessed = preprocess_mel(mel)
    mel_preprocessed = np.expand_dims(mel_preprocessed, axis=0)  # Añadir batch

    # 3. Predicción
    pred = model.predict(mel_preprocessed)
    predicted_class = CLASSES[np.argmax(pred)]

    return predicted_class
