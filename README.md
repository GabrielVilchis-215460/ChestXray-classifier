# ⚕️ Detector de Neumonía en Radiografías

Aplicación web con FastAPI + HTML que analiza radiografías de tórax usando
un modelo EfficientNetB0 entrenado con Transfer Learning.

## Estructura del proyecto

```
├── main.py               ← API FastAPI
├── requirements.txt      ← Dependencias
├── templates/
│   └── index.html        ← Frontend
├── static/               ← Archivos estáticos
```

## Setup

### 1. Clonar / copiar el proyecto

```bash
cd ChestXray-classifier/
```

### 2. Crear entorno virtual

```bash
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
http://localhost:8000
```

## Funcionalidades

- **Subida de imagen** con drag & drop o selector de archivos
- **Predicción** con EfficientNetB0 (NORMAL / PNEUMONIA)
- **Confianza** y barras de probabilidad por clase
- **6 visualizaciones diagnósticas:**
  - Original
  - Mapa de calor Grad-CAM (zonas que activaron la predicción)
  - Escala de grises
  - Detección de bordes (Canny)
  - Contraste mejorado (CLAHE)
  - Negativo / inversión de colores
- **Recomendaciones clínicas** según el resultado
- **Lightbox** para ver cada visualización en grande
- **Disclaimer** médico
