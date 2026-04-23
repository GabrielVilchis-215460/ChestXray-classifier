import io, base64, cv2, os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

app = FastAPI(title="Chest X-Ray Pneumonia Detector") # esto puede cambiar

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Cargar el modelo
model = tf.keras.models.load_model("modelo_chestxray-v2.keras")

# Configuraciones del modelo
IMG_SIZE      = 224
CLASS_NAMES   = ["NORMAL", "PNEUMONIA"]
LAST_CONV     = None   # se detecta automáticamente

# ── Utilidades de imagen ─

def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def arr_to_b64(arr: np.ndarray, fmt="PNG") -> str:
    """Convierte ndarray HxWxC (0-255 uint8) o HxW a base64 PNG."""
    img = Image.fromarray(arr.astype(np.uint8))
    return pil_to_b64(img, fmt)

def preprocess(img_pil: Image.Image) -> np.ndarray:
    img = img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    return preprocess_input(arr)

# Grad-CAM

def get_last_conv_name(mdl) -> str:
    global LAST_CONV
    if LAST_CONV:
        return LAST_CONV
    base = mdl.get_layer("efficientnetb0")
    LAST_CONV = [l.name for l in base.layers if "conv" in l.name.lower()][-1]
    return LAST_CONV

def make_gradcam(img_array: np.ndarray, mdl, pred_index: int) -> np.ndarray:
    base      = mdl.get_layer("efficientnetb0")
    conv_name = get_last_conv_name(mdl)
    inner     = tf.keras.Model(inputs=base.input,
                               outputs=base.get_layer(conv_name).output)
    with tf.GradientTape() as tape:
        conv_out = inner(img_array)
        tape.watch(conv_out)
        x = conv_out
        for layer in mdl.layers[1:]:
            x = layer(x)
        class_score = x[:, pred_index]
    grads       = tape.gradient(class_score, conv_out)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap     = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def heatmap_overlay(img_rgb: np.ndarray, heatmap: np.ndarray, alpha=0.45) -> np.ndarray:
    """Superpone Grad-CAM sobre imagen RGB (0-255)."""
    img_f   = img_rgb.astype(np.float32) / 255.0
    jet     = cm.get_cmap("jet")
    jet_arr = jet(np.arange(256))[:, :3]
    colored = jet_arr[np.uint8(255 * heatmap)]
    resized = np.array(Image.fromarray(np.uint8(colored * 255))
                       .resize((img_rgb.shape[1], img_rgb.shape[0]))) / 255.0
    overlay = resized * alpha + img_f * (1 - alpha)
    return np.uint8(overlay * 255)

# Filtros de visualización ──────────────────────────────────────────────────
def apply_filters(img_pil: Image.Image, heatmap: np.ndarray):
    """Devuelve dict con todas las visualizaciones en base64."""
    img_rgb = np.array(img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Original
    original_b64 = arr_to_b64(img_rgb)

    # 2. Grad-CAM
    gradcam_b64 = arr_to_b64(heatmap_overlay(img_rgb, heatmap))

    # 3. Escala de grises
    gray_b64 = arr_to_b64(np.stack([img_gray]*3, axis=-1))

    # 4. Canny — bordes resaltados
    edges      = cv2.Canny(img_gray, threshold1=40, threshold2=120)
    edges_rgb  = np.stack([edges]*3, axis=-1)
    canny_b64  = arr_to_b64(edges_rgb)

    # 5. CLAHE — contraste mejorado
    clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_gray   = clahe.apply(img_gray)
    clahe_rgb    = np.stack([clahe_gray]*3, axis=-1)
    clahe_b64    = arr_to_b64(clahe_rgb)

    # 6. Negativo / inversión
    inverted     = 255 - img_rgb
    inverted_b64 = arr_to_b64(inverted)

    return {
        "original":  original_b64,
        "gradcam":   gradcam_b64,
        "grayscale": gray_b64,
        "canny":     canny_b64,
        "clahe":     clahe_b64,
        "inverted":  inverted_b64,
    }

# Sugerencias clínicas

def build_suggestions(label: str, confidence: float) -> list[str]:
    suggestions = []
    if label == "PNEUMONIA":
        suggestions += [
            "Consultar a un médico especialista de inmediato.",
            "Realizar pruebas complementarias: hemograma, PCR, cultivo de esputo.",
            "Evaluar saturación de oxígeno y signos vitales.",
            "Considerar tratamiento antibiótico o antiviral según etiología.",
        ]
        if confidence < 0.75:
            suggestions.append(
                "Confianza moderada — se recomienda segunda opinión o nueva radiografía."
            )
    else:
        suggestions += [
            "Radiografía sin signos evidentes de neumonía.",
            "Mantener seguimiento clínico si el paciente presenta síntomas.",
            "Repetir estudio si los síntomas persisten o empeoran.",
        ]
        if confidence < 0.80:
            suggestions.append(
                "Confianza moderada — no descarte patología con sola esta imagen."
            )
    return suggestions

# ── Rutas ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validar tipo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    contents = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen.")

    mdl        = model
    arr        = preprocess(img_pil)
    arr_batch  = np.expand_dims(arr, axis=0)

    # Predicción
    preds      = mdl.predict(arr_batch, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    label      = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])
    probs      = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}

    # Grad-CAM
    heatmap = make_gradcam(arr_batch, mdl, pred_idx)

    # Filtros
    visuals = apply_filters(img_pil, heatmap)

    return JSONResponse({
        "label":       label,
        "confidence":  round(confidence * 100, 2),
        "probs":       {k: round(v * 100, 2) for k, v in probs.items()},
        "suggestions": build_suggestions(label, confidence),
        "visuals":     visuals,
    })
