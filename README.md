# cell-live-dead-classifier
CNN classifier for distinguishing live vs. dead cells using microscopy images. Includes training, prediction, and dataset preparation.
---
🔬 Clasificador de Células Vivas y Muertas con Redes Neuronales (CNN)

Este proyecto utiliza visión por computadora y redes neuronales convolucionales (en inglés, CNN) para clasificar imágenes microscópicas de células como vivas o muertas.

Las CNN es como una serie de filtros o lupas que miran una imagen y van detectando:

bordes,

texturas,

formas,

patrones repetidos,

regiones importantes.

El objetivo es demostrar cómo la inteligencia artificial puede analizar imágenes biomédicas de forma automática.

Se incluye:

Script para entrenar un modelo desde cero

Script para hacer predicciones

Guía completa para usarlo en Visual Studio Code (VS Code)

Dataset público utilizado

Explicación clara del funcionamiento del sistema
---
📁 1. Dataset Utilizado

Este proyecto utiliza el dataset público:

👉 Cell-Dataset – por Max Kudi
🔗 https://github.com/maxKudi/Cell-Dataset

Contiene imágenes reales de células clasificadas como:

live/ → células vivas

dead/ → células muertas

Para usarlo en este proyecto se reorganizó de la siguiente forma:
```
data/
 ├── train/
 │     ├── dead/
 │     └── live/
 └── val/
       ├── dead/
       └── live/
```

Esto permite que TensorFlow asigne:

Clase 0 → dead

Clase 1 → live
---
🧠 2. Arquitectura del Modelo

📌 1. Rescaling (normalización)

Antes de analizar la imagen, se normalizan sus valores a un rango entre 0 y 1.
Esto ayuda a que la red aprenda de forma más estable.

📌 2. Capas Convolucionales + MaxPooling2D

Estas son las capas principales de la CNN:

🔹 Conv2D (32 filtros)

Busca patrones simples como líneas o bordes.

🔹 MaxPooling2D

Reduce el tamaño de la imagen para quedarse solo con la información más importante.

Este bloque se repite tres veces, aumentando la cantidad de filtros:

Conv2D (32 filtros) → patrones simples  
Conv2D (64 filtros) → patrones más complejos  
Conv2D (128 filtros) → detalles avanzados de estructuras celulares


Cada capa se vuelve más “inteligente”, detectando patrones más específicos de las células.

📌 3. Flatten

Transforma la información de las imágenes (que es 2D) en un vector (1D) para poder enviarlo a las capas finales de decisión.

📌 4. Dense(128) + ReLU

Una capa totalmente conectada que funciona como el “juez” principal.
Aquí la red combina todos los patrones aprendidos para tomar decisiones.

ReLU es una función que ayuda a que el modelo aprenda relaciones más complejas.

📌 5. Dropout (0.5)

Apaga aleatoriamente el 50% de las neuronas en esta capa durante el entrenamiento.

Esto evita que el modelo “memorice” las imágenes y lo obliga a generalizar mejor.

📌 6. Dense(1) + Sigmoid → salida final

Esta es la capa más importante.

Tiene 1 sola neurona, porque solo necesitamos una respuesta:
¿Está viva la célula? Sí o no.

La función Sigmoid convierte la salida en un número entre 0 y 1.

Ese número se interpreta como:

→ Probabilidad de que la célula esté viva

Ejemplos:

0.85 → 85% de probabilidad de ser “viva”

0.10 → 10% de probabilidad de ser “viva” (es decir, está muerta)

0.50 → la red no está segura

📊 Características del modelo
✔ binary_crossentropy como función de pérdida

Sirve para problemas donde solo hay dos clases: viva / muerta.

✔ adam como optimizador

Es un algoritmo que ajusta los parámetros del modelo para que aprenda lo más rápido y estable posible.

✔ Tamaño de imagen: 128×128 px

Todas las imágenes se redimensionan a este tamaño antes de entrenar.

✔ Salida del modelo

Un número entre 0 y 1 que representa la probabilidad de que la célula esté viva.
---
🏋️‍♂️ 3. Entrenamiento del Modelo

El entrenamiento se realiza con el archivo:

```
cnn_ldc.py

```

Para entrenarlo:

```

python cnn_ldc.py

```

El script:

Carga las imágenes del dataset

Aplica data augmentation (rotación, zoom, flip)

Normaliza las imágenes (con Rescaling(1/255))

Entrena durante 15 épocas

Guarda el modelo como:

modelo_ldc_live_dead.h5

---

🔍 4. Predicción sobre Nuevas Imágenes

Para clasificar una imagen se usa:

predict_ldc.py


Ejemplo:

python predict_ldc.py


Salida esperada:

Imagen: 2.png
Predicción: live
Confianza: 0.91

🔥 Importante

El modelo ya normaliza internamente las imágenes con Rescaling(1/255).
Por lo tanto, NO debes normalizar nuevamente en la fase de predicción.
(Este fue un bug original que ya está corregido.)

---
🧠 Diagrama de la Arquitectura del Modelo (explicación visual)


                        ┌──────────────────────────┐
                        │     Imagen de entrada    │
                        │        128 × 128 × 3     │
                        └────────────┬─────────────┘
                                     │
                                     ▼
                       ┌──────────────────────────┐
                       │   Rescaling (normaliza)  │
                       │   Valores 0–255 → 0–1    │
                       └────────────┬─────────────┘
                                     │
                                     ▼
          ┌────────────────────── Primera etapa de aprendizaje ───────────────────────┐
          │                                                                           │
          │   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐              │
          │   │   Conv2D     │     │   Conv2D     │     │   Conv2D     │              │
          │   │ (32 filtros) │ →   │ (64 filtros) │ →   │ (128 filtros)│              │
          │   └──────┬───────┘     └──────┬───────┘     └──────┬──────┘               │
          │          │                    │                    │                      │
          │   ┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐                 │
          │   │ MaxPooling  │     │ MaxPooling  │     │ MaxPooling  │                 │
          │   └─────────────┘     └─────────────┘     └─────────────┘                 │
          │                                                                           │
          └───────────────────────────────────────────────────────────────────────────┘

                                     │
                                     ▼
                           ┌────────────────┐
                           │    Flatten     │
                           │ (pasa a 1D)    │
                           └───────┬────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  Dense (128) + ReLU    │
                       │ Toma decisiones        │
                       └───────┬────────────────┘
                               │
                               ▼
                       ┌────────────────────────┐
                       │   Dropout (0.5)        │
                       │ Evita sobreajuste      │
                       └───────┬────────────────┘
                               │
                               ▼
                ┌───────────────────────────────────────────┐
                │          Dense (1) + Sigmoid              │
                │   Salida entre 0 y 1 = probabilidad de    │
                │         que la célula esté viva           │
                └───────────────────────────────────────────┘

                                     │
                                     ▼
                        ┌──────────────────────────┐
                        │  Resultado final         │
                        │  Ej: 0.87 = 87% viva     │
                        └──────────────────────────┘

