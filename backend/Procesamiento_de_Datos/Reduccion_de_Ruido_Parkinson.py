import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

# Carpetas
input_folder = "./backend/dataset/PD_AH/"
output_folder = "./backend/dataset/Procesados_Parkinson/"
os.makedirs(output_folder, exist_ok=True)

# Funciones auxiliares
def normalize_audio(y):
    return y / np.max(np.abs(y))

def segmentar_audio(y, sr, duracion=3):
    """Divide un audio en fragmentos de X segundos"""
    muestras_segmento = duracion * sr
    segmentos = []
    for i in range(0, len(y), muestras_segmento):
        seg = y[i:i+muestras_segmento]
        if len(seg) == muestras_segmento:
            segmentos.append(seg)
    return segmentos

# Procesamiento por lote
for filename in os.listdir(input_folder):
    if filename.endswith(".wav"):
        path_in = os.path.join(input_folder, filename)
        print(f"🎧 Procesando: {filename}")

        # Cargar audio
        y, sr = librosa.load(path_in, sr=None)

        # 1️⃣ Eliminar ruido
        y_clean = nr.reduce_noise(y=y, sr=sr)

        # 2️⃣ Normalizar volumen
        y_norm = normalize_audio(y_clean)

        # 3️⃣ Segmentar audios largos (3 segundos)
        segmentos = segmentar_audio(y_norm, sr, duracion=3)

        # 4️⃣ Guardar los segmentos
        base_name = os.path.splitext(filename)[0]
        for i, seg in enumerate(segmentos):
            path_out = os.path.join(output_folder, f"{base_name}_seg{i}.wav")
            sf.write(path_out, seg, sr)

        print(f"✅ {filename} procesado ({len(segmentos)} segmentos creados)\n")

print("🎯 ¡Todos los audios han sido procesados!")
