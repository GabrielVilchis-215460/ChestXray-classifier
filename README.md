# ⚕️ PneumoScan — Clasificador de Radiografías Torácicas

Aplicación web con **FastAPI + HTML** que analiza radiografías de tórax y clasifica entre **COVID-19, Non-COVID (neumonía bacteriana/viral) y Normal**, usando un modelo EfficientNetB0 entrenado con Transfer Learning en Google Colab.

---

## 🗂️ Estructura del proyecto

```
ChestXray-classifier/
├── main.py                          ← API FastAPI
├── requirements.txt                 ← Dependencias
├── modelo_chestxray-covid-v2.keras  ← Modelo entrenado
├── static/
│   ├── index.html                   ← Detector (página principal)
│   ├── guide.html                   ← Guía radiológica
│   └── about.html                   ← Acerca del modelo
```

---

## 🧠 Modelo

| Parámetro            | Valor                                                      |
| -------------------- | ---------------------------------------------------------- |
| Arquitectura base    | EfficientNetB0 (ImageNet)                                  |
| Clases               | COVID-19 · Non-COVID · Normal                              |
| Exactitud en test    | ~88%                                                       |
| Imagen de entrada    | 224 × 224 × 3                                              |
| Capas personalizadas | GlobalAvgPool → Dense 256 (ReLU) → Dropout 0.3 → Softmax 3 |

### Dataset — COVID-QU-Ex

Descargado desde Kaggle (`anasmohammedtahir/covidqu`), partición **Infection Segmentation Data**. Se reorganizó la estructura y se balancearon las clases eliminando imágenes de la clase mayoritaria (COVID-19):

| Split | COVID-19 | Non-COVID | Normal | Total |
| ----- | -------- | --------- | ------ | ----- |
| Train | 932      | 932       | 932    | 2,796 |
| Test  | 291      | 291       | 291    | 873   |
| Val   | 233      | 233       | 233    | 699   |

_Link del dataset en Kaggle_: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu/data

### Entrenamiento en dos fases

**Fase 1 — Transfer Learning** (capas base congeladas)

- Learning rate: `0.0001`
- Épocas: 15 (early stopping, patience 5)
- Optimizador: Adam
- Loss: Categorical Crossentropy

**Fase 2 — Fine-Tuning** (últimas 10 capas descongeladas)

- Learning rate: `1e-5`
- Épocas: 15 (early stopping, patience 4)
- Se mantienen los class weights balanceados

### Data Augmentation

```python
ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.15
)
```

---

## 🚀 Setup local

### 1. Requisitos previos

- Python 3.11 (recomendado — TensorFlow no soporta 3.12)
- GPU NVIDIA opcional pero recomendada para inferencia rápida

### 2. Crear entorno virtual

```bash
cd ChestXray-classifier/
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Levantar el servidor

```bash
uvicorn main:app --reload
```

### 5. Abrir en el navegador

```
http://127.0.0.1:8000
```

---

## ✨ Funcionalidades

### Detector (`/static/index.html`)

- Subida de imagen con **drag & drop** o selector de archivos
- Predicción con EfficientNetB0 (3 clases)
- Veredicto visual con color según resultado
- Barras de probabilidad por clase
- **6 visualizaciones diagnósticas** en grid con lightbox:
  - Original
  - Mapa de calor **Grad-CAM** (zonas que activaron la predicción)
  - **Realce de texturas** (unsharp masking)
  - Detección de bordes (Canny)
  - Contraste mejorado (CLAHE)
  - Negativo / inversión de colores
- Recomendaciones clínicas según resultado y nivel de confianza
- Disclaimer médico

### Guía Radiológica (`/static/guide.html`)

- Explicación visual de cómo identificar cada condición en radiografías
- Diagramas SVG de anatomía pulmonar
- Tabla comparativa de características entre las 3 clases

### Acerca del Modelo (`/static/about.html`)

- Métricas de rendimiento
- Diagrama de arquitectura
- Parámetros de entrenamiento
- Limitaciones conocidas del modelo
- Stack tecnológico

---

## 🔑 Obtener el kaggle.json

El notebook descarga el dataset directamente desde Kaggle. Para eso necesitas tu archivo de credenciales `kaggle.json`:

1. Entra a [kaggle.com](https://www.kaggle.com) e inicia sesión
2. Ve a tu perfil → **Settings**
3. Baja hasta la sección **API**, luego **Legacy API Credentials** y haz clic en **Create Legacy API Key**
4. Se descargará automáticamente el archivo `kaggle.json`
5. Al correr el notebook, la celda `files.upload()` te pedirá que lo subas — selecciona ese archivo

```python
# Esta celda en el notebook sube tu kaggle.json a Colab
files.upload()  # ← selecciona el kaggle.json descargado
```

---

## 🔬 Análisis realizado en el notebook

El notebook `Topicos-proyecto-vFINAL.ipynb` incluye:

1. Descarga automática del dataset desde Kaggle
2. Reorganización y balanceo de clases
3. Análisis exploratorio: histograma de intensidad de píxeles por clase y ejemplos visuales
4. Entrenamiento en dos fases con EfficientNetB0
5. Reporte de clasificación con macro average
6. Visualización de predicciones (verde = acierto, rojo = error)
7. Grad-CAM sobre el conjunto de prueba
8. Matriz de confusión absoluta y normalizada
9. Curvas ROC con AUC por clase
10. Guardado del modelo en Google Drive

---

## ⚠️ Aviso

Este proyecto fue desarrollado con fines **académicos**. Los resultados no reemplazan el diagnóstico médico profesional. Cualquier decisión clínica debe ser tomada por un médico calificado.
