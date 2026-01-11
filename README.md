# 🎙️ Detector de Acentos con Deep Learning (VGG16)

Este proyecto utiliza una red neuronal convolucional basada en la arquitectura **VGG16** para clasificar acentos del idioma español en tres categorías: **Andino**, **Mexicano** y **Español**. La interfaz gráfica está construida con **Gradio**.

## 🚀 Características
- [cite_start]**Arquitectura**: Transfer Learning utilizando VGG16[cite: 260].
- [cite_start]**Procesamiento**: Transformación de audio a Espectrogramas de Mel de $224 \times 224$[cite: 146, 147].
- **Interfaz**: Permite grabar voz en tiempo real o subir archivos `.wav`/`.mp3`.
- [cite_start]**Clasificación**: Salida probabilística para las 3 clases entrenadas[cite: 274].

## 📊 Rendimiento del Modelo
[cite_start]Tras un proceso de *Fine-Tuning*[cite: 486], el modelo alcanzó los siguientes resultados en el set de prueba (Test):
* [cite_start]**Precisión General (Accuracy)**: 66.21%[cite: 580].
* [cite_start]**Acento con mejor desempeño**: México (F1-score: 0.79)[cite: 584].
* [cite_start]**Acento Andino**: F1-score de 0.58[cite: 584].
* [cite_start]**Acento España**: F1-score de 0.63[cite: 584].

## 🛠️ Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone <tu-url-del-repo>
   cd ia_audio ```

2. **Instalar dependencias:**
  ```bash
   pip install -r requirements.txt
  ```

3. **Ejecutar la aplicación:**  
   ```bash
    python gradio2.py