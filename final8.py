import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
import os

# --- 1. DEFINICIÓN DE LA ARQUITECTURA V6 ---
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
model_path = 'mejor_modelo_acentos_vgg16_imagenet_v7.h5'
model = build_model_v6()

try:
    model.load_weights(model_path)
    print("✅ ESTRUCTURA Y PESOS V6 CARGADOS CON ÉXITO")
except Exception as e:
    print(f"❌ Error al cargar: {e}")

mapping = {0: 'Andino', 1: 'México', 2: 'España'}
uploaded_files_cache = {} # Diccionario para gestionar la galería

# --- 3. FUNCIÓN DE PREDICCIÓN (CON PRE-ÉNFASIS Y SEGMENTACIÓN) ---
def predict_accent(audio_path):
    if audio_path is None: return None
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        yt, _ = librosa.effects.trim(y, top_db=20)

        # Segmentación para manejar audios largos
        samples_per_seg = int(2.56 * 16000)
        if len(yt) < samples_per_seg:
            segments = [np.pad(yt, (0, samples_per_seg - len(yt)))]
        else:
            segments = [yt[i:i + samples_per_seg] for i in range(0, len(yt)-samples_per_seg+1, samples_per_seg)]
        all_preds = []

        for seg in segments:
            # Pre-énfasis manual
            seg_pre = np.append(seg[0], seg[1:] - 0.97 * seg[:-1])
            S = librosa.feature.melspectrogram(
                y=seg_pre, sr=sr, n_fft=1024, hop_length=183, n_mels=224,
                window='hamming', fmin=20, fmax=8000
            )
            S_db = librosa.power_to_db(S, top_db=80)
       
            # Normalización Min-Max
            S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
            img = np.stack([S_norm] * 3, axis=-1)
            img = np.expand_dims(img, axis=0)
            all_preds.append(model.predict(img, verbose=0)[0])

        avg_preds = np.mean(all_preds, axis=0)
        return {mapping[i]: float(avg_preds[i]) for i in range(3)}
    except Exception as e:
        return {"Error": str(e)}

# --- 4. LÓGICA DE LA GALERÍA ---
def update_gallery(files):
    global uploaded_files_cache
    if files is None: return gr.update(choices=[]), "Galería vacía"
    uploaded_files_cache = {os.path.basename(f.name): f.name for f in files}
    nombres = list(uploaded_files_cache.keys())
    return gr.update(choices=nombres, value=nombres[0] if nombres else None), f"✅ {len(nombres)} audios en galería"


def analyze_from_gallery(selected_name):
    if not selected_name: return None
    return predict_accent(uploaded_files_cache[selected_name])

# --- 5. INTERFAZ DE GRADIO ACTUALIZADA ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Detector de Acentos Pro V6")

    with gr.Tab("Prueba Individual (Micro/Upload)"):
        with gr.Row():
            with gr.Column():
                audio_single = gr.Audio(type="filepath", label="Grabación directa", sources=["microphone", "upload"])
                btn_run_single = gr.Button("🚀 Analizar Audio Actual", variant="primary")
            with gr.Column():
                label_single = gr.Label(label="Resultado")
    with gr.Tab("Galería de Archivos (Carga Masiva)"):
        with gr.Row():
            with gr.Column():
                file_bulk = gr.File(file_count="multiple", label="Sube múltiples archivos")
                gallery_status = gr.Markdown("Esperando archivos...")

                # Dropdown para seleccionar
                selector = gr.Dropdown(label="Selecciona un audio de la lista", choices=[])

                # NUEVO: Reproductor para escuchar el archivo seleccionado
                audio_player = gr.Audio(label="Escuchar selección", interactive=False)
                btn_run_gallery = gr.Button("🚀 Analizar de la Galería", variant="primary")
            with gr.Column():
                label_gallery = gr.Label(label="Resultado de Selección")

    # --- LÓGICA DE EVENTOS ---

    # Al subir archivos a la galería
    file_bulk.change(fn=update_gallery, inputs=file_bulk, outputs=[selector, gallery_status])

    # NUEVO: Al cambiar la selección en el Dropdown, se actualiza el reproductor
    def load_to_player(selected_name):
        if selected_name in uploaded_files_cache:
            return uploaded_files_cache[selected_name]
        return None
    selector.change(fn=load_to_player, inputs=selector, outputs=audio_player)
    # Botones de análisis
    btn_run_single.click(fn=predict_accent, inputs=audio_single, outputs=label_single)
    btn_run_gallery.click(fn=analyze_from_gallery, inputs=selector, outputs=label_gallery)



if __name__ == "__main__":
    demo.launch(share=True)