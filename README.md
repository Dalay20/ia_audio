# ia_audio

## Pasos para conectar el modelo y estructura del proyecto

1. **Estructura de carpetas**
   - El modelo entrenado debe estar en `model/accent_mobilenetv2.keras`.
   - El código principal está en la raíz (`app.py`, `inference.py`).
   - Las utilidades de audio y preprocesamiento están en `utils/audio.py` y `utils/preprocess.py`.

2. **Carga del modelo**
   - El modelo se carga en el código así:
     ```python
     MODEL_PATH = "model/accent_mobilenetv2.keras"
     model = tf.keras.models.load_model(MODEL_PATH)
     ```

3. **Procesamiento del audio**
   - Convierte el archivo `.wav` a Mel spectrogram con:
     ```python
     from utils.audio import audio_to_mel
     mel = audio_to_mel("ruta/del/audio.wav")
     ```
   - Preprocesa el Mel para MobileNetV2:
     ```python
     from utils.preprocess import preprocess_mel
     mel = preprocess_mel(mel)
     mel = np.expand_dims(mel, axis=0)
     ```

4. **Predicción**
   - Realiza la predicción:
     ```python
     pred = model.predict(mel)
     pred_class = CLASSES[np.argmax(pred)]
     confidence = np.max(pred)
     ```

5. **Interfaz y uso**
   - Usa `app.py` para una interfaz web con Streamlit.
   - Usa `inference.py` para predicción desde scripts:
     ```python
     from inference import predict_accent
     resultado = predict_accent("ruta/del/audio.wav")
     ```

---

### Resumen del flujo
1. El usuario sube o indica un archivo `.wav`.
2. El audio se convierte y preprocesa a Mel spectrogram.
3. El modelo predice el acento usando el archivo en `model/`.
4. El resultado se muestra en la interfaz o se retorna desde la función.