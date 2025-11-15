from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ====== 1. RUTAS Y PARÁMETROS BÁSICOS ======

BASE_DIR = Path(__file__).resolve().parent

data_dir_train = BASE_DIR / "data" / "train"
data_dir_val = BASE_DIR / "data" / "val"

img_height = 128
img_width = 128
batch_size = 32

# ====== 2. CARGAR LOS DATOS DESDE CARPETAS ======

train_ds = keras.utils.image_dataset_from_directory(
    data_dir_train,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"   # salida 0/1
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir_val,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="binary"
)

class_names = train_ds.class_names
print("Clases detectadas:", class_names)  # debería ser ['dead', 'live'] (o similar)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# ====== 3. DATA AUGMENTATION (para evitar overfitting) ======

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ====== 4. DEFINIR LA RED NEURONAL (CNN) ======

model = keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")  # 1 neurona → probabilidad de "live"
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ====== 5. ENTRENAR EL MODELO ======

epochs = 15

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# ====== 6. GUARDAR EL MODELO ENTRENADO ======

model_path = BASE_DIR / "modelo_ldc_live_dead.h5"
model.save(model_path)
print(f"✅ Modelo guardado en: {model_path}")
