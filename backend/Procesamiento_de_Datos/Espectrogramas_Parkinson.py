import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

input_folder = "./backend/dataset/Procesados_Parkinson/"
output_folder = "./backend/dataset/espectrogramas_parkinson/"
os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        file_path = os.path.join(input_folder, file)
        y, sr = librosa.load(file_path, sr=16000)

        # Generar espectrograma Mel
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Guardar imagen
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
        plt.axis('off')
        plt.tight_layout()
        out_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"✅ Espectrograma guardado: {out_path}")

print("🎯 Todos los audios fueron transformados en espectrogramas.")
