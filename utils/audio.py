# utils/audio.py
import numpy as np
import librosa

def audio_to_mel(audio_path, sr=16000, n_mels=128, n_fft=1024, hop_length=512, duration=3):
    """
    Convierte un archivo de audio .wav a un Mel spectrogram de forma estándar.

    Args:
        audio_path (str): Ruta del archivo de audio.
        sr (int): Sample rate para cargar el audio.
        n_mels (int): Número de bandas mel.
        n_fft (int): Tamaño del FFT.
        hop_length (int): Hop length.
        duration (float): Duración máxima del audio en segundos (recorta o rellena).

    Returns:
        np.ndarray: Mel spectrogram en forma de matriz 2D (n_mels x frames).
    """
    # Cargar audio
    y, _ = librosa.load(audio_path, sr=sr)

    # Recortar o rellenar para que tenga duración fija
    target_length = int(sr * duration)
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, max(0, target_length - len(y))), 'constant')

    # Calcular Mel spectrogram
    mel = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convertir a escala logarítmica (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db
