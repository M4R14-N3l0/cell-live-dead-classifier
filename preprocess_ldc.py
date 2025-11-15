import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------
# CONFIGURACIÓN DEL PROYECTO
# ---------------------------------------------------------

# Carpeta donde está este script
BASE_DIR = Path(__file__).resolve().parent

# Carpeta con los .png y .xml
ORIGINAL_DIR = BASE_DIR / "Original Dataset"

# Carpeta donde se guardarán los parches para clasificación
OUT_BASE = BASE_DIR / "data"

# 80% train, 20% validación
TRAIN_RATIO = 0.8

# Tamaño de los parches de cada célula
PATCH_SIZE = (128, 128)

# ---------------------------------------------------------
# CREAR CARPETAS DE SALIDA
# ---------------------------------------------------------

for split in ["train", "val"]:
    for cls in ["live", "dead"]:
        (OUT_BASE / split / cls).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# LISTAR LOS XML
# ---------------------------------------------------------

xml_files = list(ORIGINAL_DIR.glob("*.xml"))
random.shuffle(xml_files)

print(f"📄 Archivos XML encontrados: {len(xml_files)}")

# ---------------------------------------------------------
# PROCESAMIENTO DE CADA XML
# ---------------------------------------------------------

for i, xml_path in enumerate(xml_files):
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        print(f"[ERROR] No se pudo leer {xml_path.name}: {e}")
        continue

    root = tree.getroot()

    # Intentar obtener el nombre del archivo desde <filename>
    filename_tag = root.find("filename")
    if filename_tag is not None:
        img_name = filename_tag.text
        img_path = ORIGINAL_DIR / img_name
    else:
        # Si no existe <filename>, asumimos mismo nombre pero .png
        img_path = xml_path.with_suffix(".png")

    if not img_path.exists():
        print(f"[AVISO] Imagen no encontrada para {xml_path.name}: {img_path}")
        continue

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] No se pudo abrir {img_path.name}: {e}")
        continue

    # Decidir si este XML va a train o val
    split = "train" if i < TRAIN_RATIO * len(xml_files) else "val"

    # -----------------------------------------------------
    # EXTRAER CADA OBJETO (CADA CÉLULA)
    # -----------------------------------------------------
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()

        # Clasificar etiqueta
        if "living" in name:
            cls = "live"
        elif "dead" in name:
            cls = "dead"
        else:
            continue  # si no es ninguna de estas, ignorar

        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        # Recortar y redimensionar parche
        crop = img.crop((xmin, ymin, xmax, ymax))
        crop = crop.resize(PATCH_SIZE)

        # Nombre de archivo único
        out_name = f"{xml_path.stem}_{cls}_{xmin}_{ymin}.png"
        out_path = OUT_BASE / split / cls / out_name

        crop.save(out_path)

print("✅ Preprocesado completado.")
print("Parches generados en la carpeta: data/train y data/val")
