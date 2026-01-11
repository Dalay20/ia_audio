import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import os

# --- 1. DEFINICIÓN DE LA ARQUITECTURA (IDÉNTICA A TU COLAB) ---
# Esta estructura debe ser exacta para que los pesos encajen [cite: 263-274]
def build_model_v5():
    # Base VGG16 con pesos de imagenet (que luego serán sobrescritos por los tuyos) [cite: 260]
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    model = models.Sequential([
        base_model, # 
        layers.GlobalAveragePooling2D(), # [cite: 288]
        layers.Dense(512, activation='relu'), # [cite: 289]
        layers.BatchNormalization(), # [cite: 292]
        layers.Dropout(0.5), # [cite: 294]
        layers.Dense(256, activation='relu'), # [cite: 297]
        layers.Dropout(0.3), # [cite: 300]
        layers.Dense(3, activation='softmax') # Capa de salida: Andino, México, España [cite: 303]
    ])
    return model

# --- 2. CARGA DEL MODELO ---
# Asegúrate de que el nombre del archivo sea el correcto en tu carpeta
model_path = 'mejor_modelo_acentos_vgg16_imagenet_v6.h5' 

print("Construyendo esqueleto y cargando conocimiento (pesos)...")
model = build_model_v5()
try:
    model.load_weights(model_path)
    print("✅ ¡ESTRUCTURA Y PESOS CARGADOS CON ÉXITO!")
except Exception as e:
    print(f"❌ Error: No se pudo cargar el archivo. Verifica que {model_path} esté en la misma carpeta.")
    raise e

# --- 3. DICCIONARIO DE MAPEO ---
# Basado en el MelDataGenerator de tu entrenamiento [cite: 122]
mapping = {0: 'Andino', 1: 'México', 2: 'España'}

# --- 4. FUNCIÓN DE PREDICCIÓN ---
def predict_accent(audio_path):
    if audio_path is None: 
        return None
    
    try:
        # Carga y normalización de audio (simulando lo que hacía tu generador)
        y, sr = librosa.load(audio_path, sr=16000)
        yt, _ = librosa.effects.trim(y, top_db=20)
        
        # Ajustamos a la duración que espera el modelo (aprox 2.56s para 224 píxeles)
        target_len = int(2.56 * 16000)
        yt = np.pad(yt, (0, max(0, target_len - len(yt))))[:target_len]
        
        # Generación del Espectrograma Mel (224x224) [cite: 138]
        S = librosa.feature.melspectrogram(
            y=yt, sr=sr, n_fft=1024, hop_length=183, n_mels=224,
            fmin=20, fmax=8000
        )
        S_db = librosa.power_to_db(S, top_db=80)
        
        # Normalización 0-1 (como estaban tus archivos .npy) [cite: 146]
        _min, _max = S_db.min(), S_db.max()
        S_norm = (S_db - _min) / (_max - _min + 1e-6)
        
        # Convertir a 3 canales para VGG16 [cite: 147]
        img = np.stack([S_norm] * 3, axis=-1)
        img = np.expand_dims(img, axis=0) # Añadir dimensión de Batch
        
        # Predicción
        preds = model.predict(img, verbose=0)[0]
        
        # Retornar diccionario con probabilidades [cite: 584]
        return {mapping[i]: float(preds[i]) for i in range(3)}
        
    except Exception as e:
        print(f"Error en el proceso: {e}")
        return {"Error": 0.0}

# --- 5. INTERFAZ DE GRADIO (BLOCKS) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Clasificador de Acentos con VGG16")
    gr.Markdown("Este modelo fue entrenado para identificar acentos **Andinos**, **Mexicanos** y **Españoles**.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                type="filepath", 
                label="Graba o sube un audio",
                sources=["microphone", "upload"]
            )
            with gr.Row():
                btn_clear = gr.Button("🗑️ Limpiar")
                btn_run = gr.Button("🚀 Analizar", variant="primary")
        
        with gr.Column():
            label_output = gr.Label(label="Predicción del Modelo")

    # Definición de acciones
    btn_run.click(
        fn=predict_accent,
        inputs=audio_input,
        outputs=label_output
    )
    
    # El botón limpiar resetea la entrada y la salida
    btn_clear.click(
        lambda: (None, None),
        inputs=None,
        outputs=[audio_input, label_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)