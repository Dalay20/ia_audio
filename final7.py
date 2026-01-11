import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import os

# --- 1. DEFINICIÓN DE LA ARQUITECTURA ---
def build_model_v6():
    base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    model = models.Sequential([
        base_model,
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalMaxPooling2D(), 
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    return model

# --- 2. CARGA DEL MODELO ---
# Asegúrate de que el nombre del archivo sea el correcto en tu carpeta
model_path = 'mejor_modelo_acentos_vgg16_imagenet_v7.h5' 

print("Construyendo esqueleto y cargando conocimiento (pesos)...")
model = build_model_v6()
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
        # 1. CARGA (SR 16000)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 2. TRIMMING (VAD) - top_db=20
        yt, _ = librosa.effects.trim(y, top_db=20)
        
        # 3. PADDING / CROP (2.56s exactos)
        target_len = int(2.56 * 16000) # 40960 muestras
        if len(yt) < target_len:
            yt = np.pad(yt, (0, target_len - len(yt)), mode='constant')
        else:
            yt = yt[:target_len]
            
        # 4. PRE-ÉNFASIS MANUAL (Igual a tu función pre_emphasis)
        # y(t) = x(t) - 0.97 * x(t-1)
        yt = np.append(yt[0], yt[1:] - 0.97 * yt[:-1])
        
        # 5. MEL SPECTROGRAM (Parámetros exactos)
        # Usamos window='hamming' y los mismos rangos de frecuencia
        S = librosa.feature.melspectrogram(
            y=yt, sr=sr,
            n_fft=1024,
            hop_length=183,
            n_mels=224,
            window='hamming', 
            fmin=20,
            fmax=8000
        )
        
        # 6. LOG-SCALE (dB) - top_db=80
        S_db = librosa.power_to_db(S, top_db=80)
        
        # 7. AJUSTE DE DIMENSIONES (224x224)
        # Por seguridad contra errores de redondeo en frames
        if S_db.shape[1] > 224:
            S_db = S_db[:, :224]
        elif S_db.shape[1] < 224:
            S_db = np.pad(S_db, ((0,0), (0, 224 - S_db.shape[1])), 
                          mode='constant', constant_values=S_db.min())
            
        # 8. MIN-MAX NORMALIZATION [0, 1]
        _min = S_db.min()
        _max = S_db.max()
        if _max - _min > 0:
            S_norm = (S_db - _min) / (_max - _min)
        else:
            S_norm = np.zeros_like(S_db)
            
        # 9. FORMATO VGG16 (Añadir 3 canales RGB y Batch)
        img = np.stack([S_norm] * 3, axis=-1)
        img = np.expand_dims(img, axis=0)
        
        # 10. PREDICCIÓN
        preds = model.predict(img, verbose=0)[0]
        
        return {mapping[i]: float(preds[i]) for i in range(3)}
        
    except Exception as e:
        print(f"Error en el preprocesamiento: {e}")
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