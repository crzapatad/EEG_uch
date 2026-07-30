import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

nombre_archivo = "allChan_1kHz_clean.mat"
with h5py.File(nombre_archivo, "r") as datos:
    matriz_eeg = np.array(datos["allChan_clean"])
    Fs = np.array(datos["Fs"])[0][0]
    canal = 0
    señal = matriz_eeg[:, canal]
    frecuencias_psd, potencia_psd = welch(señal, fs=Fs, nperseg=4096)
    indice = np.argmax(potencia_psd)
    frecuencia_pico = frecuencias_psd[indice]
    potencia_pico = potencia_psd[indice]
    plt.figure(figsize=(10,5))
    plt.plot(frecuencias_psd, potencia_psd, color="blue")
    plt.scatter(frecuencia_pico, potencia_pico, color="red", s=80, label=f"Pico = {frecuencia_pico:.2f} Hz")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Densidad de potencia")
    plt.title(f"PSD - Canal {canal+1}")
    plt.xlim(0,100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("PSD_final_canal_1.png", dpi=300, bbox_inches="tight")
    plt.show()
    frecuencias_spec, tiempos_spec, Sxx = spectrogram(señal, fs=Fs, nperseg=1024, noverlap=512)
    plt.figure(figsize=(12,6))
    plt.pcolormesh(tiempos_spec, frecuencias_spec, 10*np.log10(Sxx + 1e-12), shading="gouraud", cmap="viridis")
    plt.colorbar(label="Potencia (dB)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Frecuencia (Hz)")
    plt.title(f"Espectrograma - Canal {canal+1}")
    plt.ylim(0,100)
    plt.tight_layout()
    plt.savefig("Espectrograma_final_canal_1.png", dpi=300, bbox_inches="tight")
    plt.show()
