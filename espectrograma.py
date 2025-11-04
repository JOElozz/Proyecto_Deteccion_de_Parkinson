import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1Cargar el audio
audio_path = "AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav"
y, sr = librosa.load(audio_path, sr=16000)  # sr = sample rate

# Calcular el espectrograma de Mel
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

#  Convertir a decibelios (escala logarítmica)      
S_dB = librosa.power_to_db(S, ref=np.max)

#  Mostrar el espectrograma
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma Mel - AH_545880204-EE87D3E2-0D4C-4EAA-ACD7-C3F177AFF62F.wav')
plt.tight_layout()
plt.show()
