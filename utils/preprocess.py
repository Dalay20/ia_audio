# utils/preprocess.py
import numpy as np
from PIL import Image

def preprocess_mel(mel, target_shape=(224, 224), to_rgb=True):
    """
    Normaliza y redimensiona un Mel spectrogram para MobileNetV2.

    Args:
        mel (np.ndarray): Mel spectrogram (2D).
        target_shape (tuple): Tamaño final (height, width) para MobileNetV2.
        to_rgb (bool): Si True, repite el canal para obtener 3 canales.

    Returns:
        np.ndarray: Mel preprocesado listo para inferencia con MobileNetV2.
    """
    # 1. Normalización estándar
    mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-6)
    mel = np.clip(mel, -3, 3)
    mel = (mel + 3) / 6  # Escalamos a [0,1]

    # 2. Resize con PIL
    mel_img = Image.fromarray((mel * 255).astype(np.uint8))
    mel_img = mel_img.resize(target_shape, Image.BILINEAR)
    mel_resized = np.array(mel_img).astype(np.float32) / 255.0

    # 3. Convertir a 3 canales si se requiere
    if to_rgb:
        mel_resized = np.repeat(mel_resized[..., np.newaxis], 3, axis=-1)

    # 4. Preprocesamiento MobileNetV2 → [-1,1]
    mel_preprocessed = (mel_resized * 2.0) - 1.0

    return mel_preprocessed
