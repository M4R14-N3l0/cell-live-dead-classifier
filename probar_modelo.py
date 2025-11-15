from pathlib import Path
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# ======== CONFIGURACIÓN ========
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_ldc_live_dead.h5"

img_height = 128
img_width = 128

# Orden de clases (debe coincidir con tus carpetas)
class_names = ["dead", "live"]   # ['dead', 'live'] si así apareció en cnn_ldc.py

# ======== CARGAR MODELO ========
model = keras.models.load_model(MODEL_PATH)

def predecir_imagen(ruta_img):
    img = image.load_img(ruta_img, target_size=(img_height, img_width))
    x = image.img_to_array(img)           # SIN /255.0
    x = np.expand_dims(x, axis=0)         # batch de 1

    prob_live = model.predict(x)[0][0]

    if prob_live >= 0.5:
        clase = "live"
        confianza = prob_live
    else:
        clase = "dead"
        confianza = 1 - prob_live

    print(f"\n🖼 Imagen: {ruta_img}")
    print(f"🔍 Predicción: {clase}")
    print(f"📊 Confianza: {confianza:.2f}\n")


# ======== PRUEBA AQUÍ ========
# Cambia esta ruta por la imagen que quieres probar
imagen_prueba = BASE_DIR / "1.png"

predecir_imagen(imagen_prueba)
