import glob
import math
import os
import re
import matplotlib
matplotlib.use("Agg")
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import welch, spectrogram, hilbert, butter, filtfilt, resample_poly

# ======================================
# Clasificación de bandas EEG
# ======================================

def clasificar_banda(f):
    if 0.5 <= f < 4:
        return "Delta"
    elif 4 <= f < 8:
        return "Theta"
    elif 8 <= f < 13:
        return "Alpha"
    elif 13 <= f < 30:
        return "Beta"
    elif 30 <= f < 50:
        return "Gamma baja"
    elif 50 <= f <= 100:
        return "Gamma alta"
    else:
        return "Fuera de banda"


def parse_open_ephys_header(header_text):
    header = {}
    for line in header_text.splitlines():
        line = line.strip()
        if not line or not line.startswith("header."):
            continue
        parts = line.split("=", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip().replace("header.", "")
        value = parts[1].strip().rstrip(";")
        if value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        header[key] = value
    return header


def read_open_ephys_continuous(path):
    with open(path, "rb") as f:
        header_bytes = f.read(1024)
        header_text = header_bytes.decode("latin1", errors="replace")
        header = parse_open_ephys_header(header_text)
        sample_rate = float(header.get("sampleRate", 0))
        bit_volts = float(header.get("bitVolts", 1.0))

        data_segments = []
        while True:
            record_header = f.read(12)
            if not record_header or len(record_header) < 12:
                break
            timestamp = int.from_bytes(record_header[0:8], "little", signed=True)
            num_samples = int.from_bytes(record_header[8:10], "little", signed=False)
            recording_number = int.from_bytes(record_header[10:12], "little", signed=False)
            samples_bytes = f.read(2 * num_samples)
            if len(samples_bytes) < 2 * num_samples:
                break
            samples = np.frombuffer(samples_bytes, dtype="<i2").astype(np.float64) * bit_volts
            marker = f.read(10)
            data_segments.append(samples)

        if len(data_segments) == 0:
            raise ValueError(f"No se pudieron leer muestras de {path}")
        data = np.concatenate(data_segments)
        return sample_rate, data, header


def subsample_open_ephys(data, original_fs, target_fs):
    if target_fs <= 0 or target_fs >= original_fs:
        return data
    gcd = math.gcd(int(original_fs), int(target_fs))
    up = target_fs // gcd
    down = original_fs // gcd
    if up == 0 or down == 0:
        raise ValueError(f"Tasas de muestreo inválidas: original={original_fs}, target={target_fs}")
    return resample_poly(data, up, down)


def load_lfp_channels(lfp_dir, target_fs=None):
    continuous_files = sorted(
        [p for p in glob.glob(os.path.join(lfp_dir, "*.continuous")) if not os.path.basename(p).startswith("._")],
        key=lambda x: int(re.search(r"CH(\d+)", os.path.basename(x), re.IGNORECASE).group(1))
    )
    if len(continuous_files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .continuous en {lfp_dir}")

    channel_data = []
    sample_rate = None
    channel_names = []
    for path in continuous_files:
        fs, data, header = read_open_ephys_continuous(path)
        if sample_rate is None:
            sample_rate = fs
        elif sample_rate != fs:
            raise ValueError(f"Frecuencias de muestreo inconsistentes en {path}")
        if target_fs is not None and target_fs < sample_rate:
            data = subsample_open_ephys(data, int(sample_rate), int(target_fs))
        channel_data.append(data)
        channel_names.append(header.get("channel", os.path.basename(path)))

    min_len = min(len(ch) for ch in channel_data)
    if any(len(ch) != min_len for ch in channel_data):
        channel_data = [ch[:min_len] for ch in channel_data]

    matriz_eeg = np.vstack(channel_data).T
    return matriz_eeg, sample_rate if target_fs is None else target_fs, channel_names


def load_mat_data(file_path):
    with h5py.File(file_path, "r") as datos:
        config = None
        if "processing" in datos and "channel_configuration" in datos["processing"]:
            config = {region: np.array(datos["processing"]["channel_configuration"][region]) for region in datos["processing"]["channel_configuration"].keys()}
        matriz_eeg = np.array(datos["allChan_clean"])
        Fs = float(np.array(datos["Fs"])[0][0])
        return matriz_eeg, Fs, config


def detect_data_source(target_fs=None):
    cwd = os.getcwd()
    mat_files = [f for f in glob.glob(os.path.join(cwd, "*.mat")) if not os.path.basename(f).startswith("._")]
    if len(mat_files) > 0:
        print("Se detectó archivo .mat. Usando la primera coincidencia disponible:")
        print(mat_files[0])
        return load_mat_data(mat_files[0])

    lfp_dir = os.path.join(cwd, "LFP")
    if os.path.isdir(lfp_dir):
        print(f"Leyendo archivos Open Ephys en {lfp_dir}")
        matriz_eeg, Fs, channel_names = load_lfp_channels(lfp_dir, target_fs=target_fs)
        return matriz_eeg, Fs, None

    raise FileNotFoundError("No se encontró un archivo .mat ni la carpeta LFP con archivos .continuous.")


def main():
    print("Iniciando lectura de datos...")
    target_fs = None
    user_target = input("Ingrese la frecuencia de muestreo deseada para el subsampleo (por ejemplo 1000), o Enter para usar la original: ").strip()
    if user_target:
        target_fs = int(user_target)
    matriz_eeg, Fs, config = detect_data_source(target_fs=target_fs)
    n_muestras = matriz_eeg.shape[0]
    n_canales = matriz_eeg.shape[1]

    print("\nInformación del registro")
    print("------------------------")
    print(f"Frecuencia de muestreo: {Fs} Hz")
    print(f"Dimensiones: {matriz_eeg.shape}")
    print(f"Número de canales: {n_canales}")
    print(f"Número de muestras: {n_muestras}")
    print(f"Duración: {n_muestras / Fs:.2f} segundos")
    print(f"Duración: {(n_muestras / Fs) / 60:.2f} minutos")

    if config is not None:
        print("\nConfiguración de canales detectada:")
        for region in config:
            contenido = np.array(config[region])
            print(f"\nRegión: {region}")
            print("Forma:", contenido.shape)
            print(contenido)

    estadisticas = []
    regiones = {
        "CPF": range(0, 10),
        "Nacc": range(10, 16),
        "Amy": range(16, 25),
        "Hyp": range(25, 32)
    }

    for canal in range(n_canales):
        señal = matriz_eeg[:, canal]
        region = ""
        for nombre, canales in regiones.items():
            if canal in canales:
                region = nombre
                break

        promedio = np.mean(señal)
        desviacion = np.std(señal)
        mediana = np.median(señal)
        mad = np.median(np.abs(señal - mediana))
        minimo = np.min(señal)
        maximo = np.max(señal)
        rango = maximo - minimo
        diff = np.diff(señal)
        diff_promedio = np.mean(diff)
        diff_std = np.std(diff)
        diff_mediana = np.median(diff)
        diff_mad = np.median(np.abs(diff - diff_mediana))
        diff_max = np.max(diff)
        diff_min = np.min(diff)
        diff_rango = diff_max - diff_min

        estadisticas.append([
            canal + 1,
            region,
            promedio,
            desviacion,
            mediana,
            mad,
            minimo,
            maximo,
            rango,
            diff_promedio,
            diff_std,
            diff_mediana,
            diff_mad,
            diff_max,
            diff_min,
            diff_rango
        ])

    df = pd.DataFrame(
        estadisticas,
        columns=[
            "Canal",
            "Región",
            "Promedio",
            "Desv_Est",
            "Mediana",
            "MAD",
            "Mínimo",
            "Máximo",
            "Máx-Mín",
            "Diff_promedio",
            "Diff_Desv_Est",
            "Diff_Mediana",
            "Diff_MAD",
            "Diff_Máximo",
            "Diff_Mínimo",
            "Diff_Máx-Mín"
        ]
    )

    print("\nEstadísticos descriptivos")
    print(df)
    df.to_csv("estadisticas_canales.csv", index=False, encoding="utf-8-sig")
    print("\nArchivo guardado como: estadisticas_canales.csv")

    EXCLUDE_29_30 = True
    exclude_set = {28, 29} if EXCLUDE_29_30 else set()

    region_avgs = {}
    for nombre, canales in regiones.items():
        canales_list = [c for c in list(canales) if c not in exclude_set and c < n_canales]
        if len(canales_list) == 0:
            region_avgs[nombre] = np.zeros(n_muestras)
        else:
            region_avgs[nombre] = np.mean(matriz_eeg[:, canales_list], axis=1)

    residual = np.zeros_like(matriz_eeg)
    for ch in range(n_canales):
        region_name = None
        for nombre, canales in regiones.items():
            if ch in canales:
                region_name = nombre
                break
        if region_name is None:
            residual[:, ch] = matriz_eeg[:, ch]
        else:
            residual[:, ch] = matriz_eeg[:, ch] - region_avgs[region_name]

    df_region = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    for nombre, vec in region_avgs.items():
        df_region[f"{nombre}_avg"] = vec
    df_region.to_csv("region_averages.csv", index=False, encoding="utf-8-sig")

    np.save("allChan_residual.npy", residual)
    with h5py.File("allChan_residual.mat", "w") as f_out:
        f_out.create_dataset("allChan_residual", data=residual)
    print("Guardados: region_averages.csv, allChan_residual.npy, allChan_residual.mat")

    segundos_plot = 5
    muestras_plot = int(segundos_plot * Fs)
    muestras_plot = min(muestras_plot, n_muestras)
    tiempo_plot = np.arange(muestras_plot) / Fs
    valid_channels = [c for c in range(n_canales) if c not in exclude_set]
    n_valid = len(valid_channels)

    plt.figure(figsize=(18,12))
    offset = 300
    colores_region = {"CPF":"red", "Nacc":"blue", "Amy":"green", "Hyp":"purple"}

    for idx, ch in enumerate(valid_channels):
        region_name = None
        for nombre, canales in regiones.items():
            if ch in canales:
                region_name = nombre
                break
        color = colores_region.get(region_name, "black")
        plt.plot(tiempo_plot, residual[:muestras_plot, ch] + idx * offset, color=color, linewidth=0.8)

    plt.yticks(np.arange(n_valid) * offset, [f"{ch+1}" for ch in valid_channels])
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Canales (residual)")
    plt.title("Actividad neuronal residual por canal (original - promedio región) - canales excluidos: 29,30")
    plt.grid(True)
    from matplotlib.lines import Line2D
    leyenda = [Line2D([0],[0], color=c, label=r) for r,c in colores_region.items()]
    plt.legend(handles=leyenda, loc="upper right")
    plt.tight_layout()
    plt.savefig("allChan_residual_plot_reduced.png", dpi=300, bbox_inches="tight")
    plt.close()

    residual_reduced = residual[:, valid_channels]
    np.save("allChan_residual_reduced.npy", residual_reduced)
    with h5py.File("allChan_residual_reduced.mat", "w") as f_out_red:
        f_out_red.create_dataset("allChan_residual_reduced", data=residual_reduced)

    original_channel_indices = valid_channels
    original_to_col = {orig + 1: idx for idx, orig in enumerate(original_channel_indices)}

    matriz_eeg = residual_reduced.copy()
    n_canales = matriz_eeg.shape[1]

    def butterworth_bandpass(signal, fs, lowcut, highcut, order=4):
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 1e-6)
        high = min(highcut / nyq, 0.999999)
        if low <= 1e-6:
            b, a = butter(order, high, btype="low")
        else:
            b, a = butter(order, [low, high], btype="bandpass")
        return filtfilt(b, a, signal)

    if 1 not in original_to_col or 17 not in original_to_col:
        raise ValueError("Los canales originales 1 o 17 no están disponibles después de la exclusión.")

    canal_1_col = original_to_col[1]
    canal_17_col = original_to_col[17]
    señal_canal_1 = matriz_eeg[:, canal_1_col]
    señal_canal_17 = matriz_eeg[:, canal_17_col]

    filtrado_canal_1 = butterworth_bandpass(señal_canal_1, Fs, 0.0, 30.0, order=4)
    filtrado_canal_17 = butterworth_bandpass(señal_canal_17, Fs, 0.0, 30.0, order=4)

    np.save("canal_1_PFC_butterworth_0_30_n4.npy", filtrado_canal_1)
    np.save("canal_17_Amy_butterworth_0_30_n4.npy", filtrado_canal_17)

    band_windows = [(i, i + 2) for i in range(0, 29)]
    df_butter_windows = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    df_butter_1 = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    df_butter_17 = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})

    for lowcut, highcut in band_windows:
        etiqueta = f"{lowcut}_{highcut}Hz"
        filtrado_1 = butterworth_bandpass(señal_canal_1, Fs, float(lowcut), float(highcut), order=4)
        filtrado_17 = butterworth_bandpass(señal_canal_17, Fs, float(lowcut), float(highcut), order=4)
        df_butter_windows[f"Canal_1_PFC_{etiqueta}"] = filtrado_1
        df_butter_windows[f"Canal_17_Amy_{etiqueta}"] = filtrado_17
        df_butter_1[f"Band_{etiqueta}"] = filtrado_1
        df_butter_17[f"Band_{etiqueta}"] = filtrado_17

    df_butter_windows.to_csv("butterworth_order4_canal_1_17_0_30_2hz.csv", index=False, encoding="utf-8-sig")
    df_butter_1.to_csv("butterworth_order4_canal_1_0_30_2hz.csv", index=False, encoding="utf-8-sig")
    df_butter_17.to_csv("butterworth_order4_canal_17_0_30_2hz.csv", index=False, encoding="utf-8-sig")

    print("Guardados: butterworth_order4_canal_1_17_0_30_2hz.csv, butterworth_order4_canal_1_0_30_2hz.csv, butterworth_order4_canal_17_0_30_2hz.csv")
    print("Guardados: versiones individuales .npy para el filtro global 0-30 Hz")

    analytic_canal_1 = hilbert(filtrado_canal_1)
    analytic_canal_17 = hilbert(filtrado_canal_17)
    fase_canal_1 = np.angle(analytic_canal_1)
    fase_canal_17 = np.angle(analytic_canal_17)
    fase_diff_17_1 = np.unwrap(fase_canal_17 - fase_canal_1)

    np.save("fase_canal_1_PFC.npy", fase_canal_1)
    np.save("fase_canal_17_Amy.npy", fase_canal_17)
    np.save("fase_diff_17_1.npy", fase_diff_17_1)

    df_phase_filtered = pd.DataFrame({
        "Tiempo_s": np.arange(n_muestras) / Fs,
        "Fase_Canal_1_PFC": fase_canal_1,
        "Fase_Canal_17_Amy": fase_canal_17,
        "Diferencia_Fase_17_1": fase_diff_17_1
    })
    df_phase_filtered.to_csv("phase_hilbert_canal_1_17_diff.csv", index=False, encoding="utf-8-sig")

    complex_vectors = np.exp(1j * fase_diff_17_1)
    mrl = np.abs(np.mean(complex_vectors))
    print(f"Phase Coherence (MRL): {mrl:.4f}")

    df_mrl = pd.DataFrame({
        "Metric": ["Phase_Coherence_MRL"],
        "Value": [mrl]
    })
    df_mrl.to_csv("phase_coherence_mrl.csv", index=False, encoding="utf-8-sig")

    print("Guardados: phase_hilbert_canal_1_17_diff.csv, phase_coherence_mrl.csv, fase_canal_1_PFC.npy, fase_canal_17_Amy.npy, fase_diff_17_1.npy")

    analytic_signal = hilbert(matriz_eeg, axis=0)
    fase_instantanea = np.angle(analytic_signal)

    np.save("allChan_phase_instantanea.npy", fase_instantanea)
    with h5py.File("allChan_phase_instantanea.mat", "w") as f_phase:
        f_phase.create_dataset("allChan_phase_instantanea", data=fase_instantanea)

    df_phase = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    for idx, origen in enumerate(original_channel_indices):
        canal_label = origen + 1
        df_phase[f"Canal_{canal_label}"] = fase_instantanea[:, idx]
    df_phase.to_csv("allChan_phase_instantanea.csv", index=False, encoding="utf-8-sig")

    def guardar_segmento_csv(nombre_archivo, inicio_s, fin_s):
        inicio = int(inicio_s * Fs)
        fin = int(fin_s * Fs)
        fin = min(fin, n_muestras)
        df_segmento = df_phase.iloc[inicio:fin].copy()
        df_segmento.to_csv(nombre_archivo, index=False, encoding="utf-8-sig")

    guardar_segmento_csv("allChan_phase_instantanea_1_6min.csv", 60, 360)
    guardar_segmento_csv("allChan_phase_instantanea_11_16min.csv", 660, 960)

    print("Guardados: allChan_phase_instantanea.npy, allChan_phase_instantanea.mat, allChan_phase_instantanea.csv")
    print("Guardados: allChan_phase_instantanea_1_6min.csv, allChan_phase_instantanea_11_16min.csv")

    n_plot_channels = min(6, n_canales)
    muestras_phase = int(min(2 * Fs, n_muestras))
    tiempo_phase = np.arange(muestras_phase) / Fs
    fase_unwrapped = np.unwrap(fase_instantanea[:muestras_phase, :n_plot_channels], axis=0)

    plt.figure(figsize=(12, 8))
    offset_phase = 10
    for i in range(n_plot_channels):
        plt.plot(tiempo_phase, fase_unwrapped[:, i] + i * offset_phase, label=f"Canal {original_channel_indices[i]+1}")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Fase instantánea (rad) + offset")
    plt.title("Fase instantánea de los primeros canales reducidos (Hilbert)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("allChan_phase_instantanea_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Guardado: allChan_phase_instantanea_plot.png")

    modo = input(
        "\n¿Qué desea analizar?\n"
        "1 -> Un canal\n"
        "2 -> Todos los canales\n\n"
        "Opción: "
    ).strip()

    if modo == "2":
        canales_a_analizar = sorted(original_to_col.keys())
        print(f"\nSe analizarán {len(canales_a_analizar)} canales (etiquetas originales, excluidos 29 y 30).")
    else:
        canal_visualizar = int(
            input("\nIngrese el canal (1-32): ")
        )
        while canal_visualizar < 1 or canal_visualizar > 32 or canal_visualizar not in original_to_col:
            canal_visualizar = int(
                input("Canal inválido o excluido. Ingrese un canal válido (1-32), no 29/30: ")
            )
        canales_a_analizar = [canal_visualizar]

    for canal_label in canales_a_analizar:
        col = original_to_col[canal_label]
        señal = matriz_eeg[:, col]
        frecuencias_psd, potencia_psd = welch(señal, fs=Fs, nperseg=4096)

        plt.figure(figsize=(10,5))
        plt.plot(frecuencias_psd, potencia_psd, color="blue", linewidth=1)
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Densidad de potencia")
        plt.title(f"PSD - Canal {canal_label} (residual)")
        plt.xlim(0,100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"PSD_Canal_{canal_label}_residual.png", dpi=300, bbox_inches="tight")
        plt.close()

        frecuencias_spec, tiempos_spec, Sxx = spectrogram(señal, fs=Fs, nperseg=1024, noverlap=512)
        plt.figure(figsize=(12,6))
        spec_db = 10 * np.log10(Sxx + 1e-12)
        plt.pcolormesh(tiempos_spec, frecuencias_spec, spec_db, shading="gouraud", cmap="viridis", vmin=-20, vmax=20)
        plt.colorbar(label="Potencia (dB)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Frecuencia (Hz)")
        plt.title(f"Espectrograma - Canal {canal_label} (residual)")
        plt.ylim(0,100)
        plt.tight_layout()
        plt.savefig(f"Espectrograma_Canal_{canal_label}_residual.png", dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(10,6))
    regiones_original = {
        "CPF": ("red", range(0,10)),
        "Nacc": ("blue", range(10,16)),
        "Amy": ("green", range(16,25)),
        "Hyp": ("purple", range(25,32))
    }

    for nombre, (color, canales_range) in regiones_original.items():
        potencias_psd = []
        for ch_idx in canales_range:
            label = ch_idx + 1
            if label not in original_to_col:
                continue
            col = original_to_col[label]
            señal = matriz_eeg[:, col]
            frecuencias_psd, potencia_psd = welch(señal, fs=Fs, nperseg=4096)
            potencias_psd.append(potencia_psd)
        if len(potencias_psd) == 0:
            continue
        promedio = np.mean(potencias_psd, axis=0)
        plt.plot(frecuencias_psd, promedio, color=color, linewidth=2, label=nombre)

    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Potencia")
    plt.title("PSD promedio por región (residual)")
    plt.xlim(0,100)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PSD_promedio_regiones_residual.png", dpi=300, bbox_inches="tight")
    plt.close()

    analisis = []
    for canal_label in sorted(original_to_col.keys()):
        col = original_to_col[canal_label]
        señal = matriz_eeg[:, col]
        frecuencias_psd, potencia_psd = welch(señal, fs=Fs, nperseg=4096)
        indice = np.argmax(potencia_psd)
        frecuencia_pico = frecuencias_psd[indice]
        potencia_pico = potencia_psd[indice]
        region = ""
        for nombre, canales_range in regiones_original.items():
            if (canal_label - 1) in canales_range:
                region = nombre
                break
        analisis.append([canal_label, region, frecuencia_pico, potencia_pico])

    df_espectral = pd.DataFrame(analisis, columns=["Canal", "Región", "Frecuencia_Pico_Hz", "Potencia_Pico"])
    df_espectral = df_espectral.round(2)
    print(df_espectral)
    df_espectral.to_csv("Analisis_Espectral.csv", index=False, encoding="utf-8-sig")
    print("\nArchivo guardado: Analisis_Espectral.csv")

    print("\nRealizando PCA...")
    X = df.drop(columns=["Canal", "Región"])
    df = df.round(1)
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)
    pca = PCA()
    componentes = pca.fit_transform(X_escalado)
    print(componentes.shape)
    print(componentes[:5, 2])
    print("\nVarianza explicada")
    for i, porcentaje in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {porcentaje*100:.2f}%")

    varianza = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Varianza (%)": pca.explained_variance_ratio_ * 100
    })
    varianza.to_csv("PCA_varianza.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_ * 100, marker="o")
    plt.xlabel("Componente principal")
    plt.ylabel("Varianza explicada (%)")
    plt.title("Scree Plot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PCA_scree_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    df_pca = pd.DataFrame({
        "Canal": df["Canal"],
        "Región": df["Región"],
        "PC1": componentes[:,0],
        "PC2": componentes[:,1],
        "PC3": componentes[:,2]
    })

    print("Entrando al gráfico 3D...")
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")
    colores = {"CPF":"red", "Nacc":"blue", "Amy":"green", "Hyp":"purple"}
    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        ax.scatter(datos_region["PC1"], datos_region["PC2"], datos_region["PC3"], color=colores[region], label=region, s=80)
        for _, fila in datos_region.iterrows():
            ax.text(fila["PC1"] + 0.03, fila["PC2"] + 0.03, fila["PC3"] + 0.03, str(int(fila["Canal"])), fontsize=9, color=colores[region])
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA tridimensional")
    ax.legend()
    plt.tight_layout()
    plt.savefig("PCA_3D.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Gráfico 3D finalizado.")

    plt.figure(figsize=(10,8))
    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        plt.scatter(datos_region["PC1"], datos_region["PC2"], color=colores[region], s=120, alpha=0.4)
        for _, fila in datos_region.iterrows():
            plt.text(fila["PC1"] + 0.03, fila["PC2"] + 0.03, str(int(fila["Canal"])), fontsize=10, fontweight="bold", color=colores[region], ha="center", va="center")

    loadings = pca.components_.T
    for i, variable in enumerate(X.columns):
        plt.arrow(0, 0, loadings[i,0]*4, loadings[i,1]*4, color="black", alpha=0.7, head_width=0.05)
        plt.text(loadings[i,0]*4.2, loadings[i,1]*4.2, variable, fontsize=9)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot del PCA")
    plt.axhline(0,color="gray",linewidth=0.5)
    plt.axvline(0,color="gray",linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("PCA_biplot.png", dpi=300, bbox_inches="tight")
    plt.close()

    df_pca.to_csv("PCA_canales.csv", index=False, encoding="utf-8-sig")
    print("\nArchivo PCA guardado correctamente.")

    plt.figure(figsize=(8,6))
    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        plt.scatter(datos_region["PC1"], datos_region["PC2"], color=colores[region], s=120, alpha=0.5, label=region)
        for _, fila in datos_region.iterrows():
            plt.text(fila["PC1"], fila["PC2"], str(int(fila["Canal"])), fontsize=10, fontweight="bold", color=colores[region], ha="center", va="center")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA de los canales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("PCA_scatter_2D.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nDetección de artefactos")
    display_channels = sorted(original_to_col.keys())
    for label in display_channels:
        col = original_to_col[label]
        señal = matriz_eeg[:, col]
        diff = np.diff(señal)
        umbral = 5 * np.std(diff)
        artefactos = np.where(np.abs(diff) > umbral)[0]
        print(f"Canal {label}: {len(artefactos)} posibles artefactos")

    segundos = 5
    muestras = int(segundos * Fs)
    muestras = min(muestras, n_muestras)
    tiempo = np.arange(muestras) / Fs
    plt.figure(figsize=(18,12))
    offset = 300
    regiones = {
        "CPF": ("red", range(0,10)),
        "Nacc": ("blue", range(10,16)),
        "Amy": ("green", range(16,25)),
        "Hyp": ("purple", range(25,32))
    }
    display_channels = sorted(original_to_col.keys())
    for idx, label in enumerate(display_channels):
        color = "black"
        for nombre_region, (col_color, canales_range) in regiones.items():
            if (label - 1) in canales_range:
                color = col_color
                break
        col = original_to_col[label]
        plt.plot(tiempo, matriz_eeg[:muestras, col] + idx*offset, color=color, linewidth=0.8)

    plt.yticks(np.arange(len(display_channels))*offset, [f"{lbl}" for lbl in display_channels])
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Canales")
    plt.title("Actividad neuronal por regiones cerebrales")
    plt.grid(True)
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0],[0],color="red",label="CPF"),
        Line2D([0],[0],color="blue",label="Nacc"),
        Line2D([0],[0],color="green",label="Amy"),
        Line2D([0],[0],color="purple",label="Hyp")
    ]
    plt.legend(handles=leyenda)
    plt.tight_layout()
    plt.savefig("allChan_activity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Guardado: allChan_activity.png")

if __name__ == "__main__":
    main()

    # Guardar matriz residual reducida (sin los canales excluidos)
    residual_reduced = residual[:, valid_channels]
    np.save("allChan_residual_reduced.npy", residual_reduced)
    with h5py.File("allChan_residual_reduced.mat", "w") as f_out_red:
        f_out_red.create_dataset("allChan_residual_reduced", data=residual_reduced)

    # -------------------------------------------------
    # Reemplazar la matriz original por la reducida para
    # el resto del flujo, pero mantener las etiquetas
    # originales de canal (no reindexar).
    # -------------------------------------------------
    original_channel_indices = valid_channels  # 0-based original indices
    # mapa original_label (1-based) -> columna en la matriz reducida
    original_to_col = {orig + 1: idx for idx, orig in enumerate(original_channel_indices)}

    # Reemplazar matriz_eeg con la matriz residual reducida
    matriz_eeg = residual_reduced.copy()
    n_canales = matriz_eeg.shape[1]

    # ======================================
    # Filtro Butterworth orden 4 para canal 1 (PFC) y canal 17 (Amy)
    def butterworth_bandpass(signal, fs, lowcut, highcut, order=4):
        nyq = 0.5 * fs
        low = max(lowcut / nyq, 1e-6)
        high = min(highcut / nyq, 0.999999)
        if low <= 1e-6:
            b, a = butter(order, high, btype="low")
        else:
            b, a = butter(order, [low, high], btype="bandpass")
        return filtfilt(b, a, signal)

    # Canales seleccionados (1 y 17 son etiquetas originales)
    canal_1_col = original_to_col[1]
    canal_17_col = original_to_col[17]

    señal_canal_1 = matriz_eeg[:, canal_1_col]
    señal_canal_17 = matriz_eeg[:, canal_17_col]

    # Filtro global de 0-30 Hz para cálculo de fase general
    filtrado_canal_1 = butterworth_bandpass(señal_canal_1, Fs, 0.0, 30.0, order=4)
    filtrado_canal_17 = butterworth_bandpass(señal_canal_17, Fs, 0.0, 30.0, order=4)

    np.save("canal_1_PFC_butterworth_0_30_n4.npy", filtrado_canal_1)
    np.save("canal_17_Amy_butterworth_0_30_n4.npy", filtrado_canal_17)

    # Crear banco de pasabanda 2 Hz solapados entre 0-30 Hz
    band_windows = [(i, i + 2) for i in range(0, 29)]
    df_butter_windows = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    df_butter_1 = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    df_butter_17 = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})

    for lowcut, highcut in band_windows:
        etiqueta = f"{lowcut}_{highcut}Hz"
        filtrado_1 = butterworth_bandpass(señal_canal_1, Fs, float(lowcut), float(highcut), order=4)
        filtrado_17 = butterworth_bandpass(señal_canal_17, Fs, float(lowcut), float(highcut), order=4)

        df_butter_windows[f"Canal_1_PFC_{etiqueta}"] = filtrado_1
        df_butter_windows[f"Canal_17_Amy_{etiqueta}"] = filtrado_17
        df_butter_1[f"Band_{etiqueta}"] = filtrado_1
        df_butter_17[f"Band_{etiqueta}"] = filtrado_17

    df_butter_windows.to_csv("butterworth_order4_canal_1_17_0_30_2hz.csv", index=False, encoding="utf-8-sig")
    df_butter_1.to_csv("butterworth_order4_canal_1_0_30_2hz.csv", index=False, encoding="utf-8-sig")
    df_butter_17.to_csv("butterworth_order4_canal_17_0_30_2hz.csv", index=False, encoding="utf-8-sig")

    print("Guardados: butterworth_order4_canal_1_17_0_30_2hz.csv, butterworth_order4_canal_1_0_30_2hz.csv, butterworth_order4_canal_17_0_30_2hz.csv")
    print("Guardados: versiones individuales .npy para el filtro global 0-30 Hz")

    # ======================================
    # Transformada de Hilbert y fase instantánea de los canales filtrados
    analytic_canal_1 = hilbert(filtrado_canal_1)
    analytic_canal_17 = hilbert(filtrado_canal_17)
    fase_canal_1 = np.angle(analytic_canal_1)
    fase_canal_17 = np.angle(analytic_canal_17)

    # Diferencia de fase (Amy - PFC)
    fase_diff_17_1 = np.unwrap(fase_canal_17 - fase_canal_1)

    np.save("fase_canal_1_PFC.npy", fase_canal_1)
    np.save("fase_canal_17_Amy.npy", fase_canal_17)
    np.save("fase_diff_17_1.npy", fase_diff_17_1)

    df_phase_filtered = pd.DataFrame({
        "Tiempo_s": np.arange(n_muestras) / Fs,
        "Fase_Canal_1_PFC": fase_canal_1,
        "Fase_Canal_17_Amy": fase_canal_17,
        "Diferencia_Fase_17_1": fase_diff_17_1
    })
    df_phase_filtered.to_csv("phase_hilbert_canal_1_17_diff.csv", index=False, encoding="utf-8-sig")

    complex_vectors = np.exp(1j * fase_diff_17_1)
    mrl = np.abs(np.mean(complex_vectors))
    print(f"Phase Coherence (MRL): {mrl:.4f}")

    df_mrl = pd.DataFrame({
        "Metric": ["Phase_Coherence_MRL"],
        "Value": [mrl]
    })
    df_mrl.to_csv("phase_coherence_mrl.csv", index=False, encoding="utf-8-sig")

    print("Guardados: phase_hilbert_canal_1_17_diff.csv, phase_coherence_mrl.csv, fase_canal_1_PFC.npy, fase_canal_17_Amy.npy, fase_diff_17_1.npy")

    # ======================================
    # Cálculo de fase instantánea por canal usando la transformada de Hilbert
    analytic_signal = hilbert(matriz_eeg, axis=0)
    fase_instantanea = np.angle(analytic_signal)

    np.save("allChan_phase_instantanea.npy", fase_instantanea)
    with h5py.File("allChan_phase_instantanea.mat", "w") as f_phase:
        f_phase.create_dataset("allChan_phase_instantanea", data=fase_instantanea)

    # Guardar fase instantánea como vectores por canal en CSV
    df_phase = pd.DataFrame({"Tiempo_s": np.arange(n_muestras) / Fs})
    for idx, origen in enumerate(original_channel_indices):
        canal_label = origen + 1
        df_phase[f"Canal_{canal_label}"] = fase_instantanea[:, idx]
    df_phase.to_csv("allChan_phase_instantanea.csv", index=False, encoding="utf-8-sig")

    # Crear CSVs de segmentos temporales específicos
    def guardar_segmento_csv(nombre_archivo, inicio_s, fin_s):
        inicio = int(inicio_s * Fs)
        fin = int(fin_s * Fs)
        fin = min(fin, n_muestras)
        df_segmento = df_phase.iloc[inicio:fin].copy()
        df_segmento.to_csv(nombre_archivo, index=False, encoding="utf-8-sig")

    guardar_segmento_csv("allChan_phase_instantanea_1_6min.csv", 60, 360)
    guardar_segmento_csv("allChan_phase_instantanea_11_16min.csv", 660, 960)

    print("Guardados: allChan_phase_instantanea.npy, allChan_phase_instantanea.mat, allChan_phase_instantanea.csv")
    print("Guardados: allChan_phase_instantanea_1_6min.csv, allChan_phase_instantanea_11_16min.csv")

    # ======================================
    # Gráfico de fase instantánea para los canales reducidos
    # ======================================
    n_plot_channels = min(6, n_canales)
    muestras_phase = int(min(2 * Fs, n_muestras))
    tiempo_phase = np.arange(muestras_phase) / Fs
    fase_unwrapped = np.unwrap(fase_instantanea[:muestras_phase, :n_plot_channels], axis=0)

    plt.figure(figsize=(12, 8))
    offset_phase = 10
    for i in range(n_plot_channels):
        plt.plot(tiempo_phase, fase_unwrapped[:, i] + i * offset_phase, label=f"Canal {original_channel_indices[i]+1}")

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Fase instantánea (rad) + offset")
    plt.title("Fase instantánea de los primeros canales reducidos (Hilbert)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("allChan_phase_instantanea_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Guardado: allChan_phase_instantanea_plot.png")

    # ======================================
    # Selección de canal(es) usando etiquetas originales
    # y generación de PSD / Espectrogramas sobre la señal residual
    # ======================================

    modo = input(
        "\n¿Qué desea analizar?\n"
        "1 -> Un canal\n"
        "2 -> Todos los canales\n\n"
        "Opción: "
    ).strip()

    # original_to_col: mapa 1-based -> columna en matriz_eeg reducida
    if modo == "2":
        canales_a_analizar = sorted(original_to_col.keys())
        print(f"\nSe analizarán {len(canales_a_analizar)} canales (etiquetas originales, excluidos 29 y 30).")
    else:
        canal_visualizar = int(
            input("\nIngrese el canal (1-32): ")
        )
        while canal_visualizar < 1 or canal_visualizar > 32 or canal_visualizar not in original_to_col:
            canal_visualizar = int(
                input("Canal inválido o excluido. Ingrese un canal válido (1-32), no 29/30: ")
            )
        canales_a_analizar = [canal_visualizar]

    # ======================================
    # PSD y espectrogramas por canal (sobre residual)
    # Usamos las etiquetas originales para títulos/archivos
    # ======================================
    for canal_label in canales_a_analizar:
        col = original_to_col[canal_label]
        señal = matriz_eeg[:, col]

        frecuencias_psd, potencia_psd = welch(
            señal,
            fs=Fs,
            nperseg=4096
        )

        plt.figure(figsize=(10,5))
        plt.plot(frecuencias_psd, potencia_psd, color="blue", linewidth=1)
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Densidad de potencia")
        plt.title(f"PSD - Canal {canal_label} (residual)")
        plt.xlim(0,100)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"PSD_Canal_{canal_label}_residual.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Espectrograma
        frecuencias_spec, tiempos_spec, Sxx = spectrogram(
            señal,
            fs=Fs,
            nperseg=1024,
            noverlap=512
        )
        plt.figure(figsize=(12,6))
        spec_db = 10 * np.log10(Sxx + 1e-12)
        plt.pcolormesh(tiempos_spec, frecuencias_spec, spec_db, shading="gouraud", cmap="viridis", vmin=-20, vmax=20)
        plt.colorbar(label="Potencia (dB)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Frecuencia (Hz)")
        plt.title(f"Espectrograma - Canal {canal_label} (residual)")
        plt.ylim(0,100)
        plt.tight_layout()
        plt.savefig(f"Espectrograma_Canal_{canal_label}_residual.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ======================================
    # PSD promedio por región (sobre residual)
    # ======================================
    plt.figure(figsize=(10,6))

    regiones_original = {
        "CPF": ("red", range(0,10)),
        "Nacc": ("blue", range(10,16)),
        "Amy": ("green", range(16,25)),
        "Hyp": ("purple", range(25,32))
    }

    for nombre, (color, canales_range) in regiones_original.items():
        potencias_psd = []
        for ch_idx in canales_range:
            label = ch_idx + 1
            if label not in original_to_col:
                continue
            col = original_to_col[label]
            señal = matriz_eeg[:, col]
            frecuencias_psd, potencia_psd = welch(señal, fs=Fs, nperseg=4096)
            potencias_psd.append(potencia_psd)
        if len(potencias_psd) == 0:
            continue
        promedio = np.mean(potencias_psd, axis=0)
        plt.plot(frecuencias_psd, promedio, color=color, linewidth=2, label=nombre)

    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Potencia")
    plt.title("PSD promedio por región (residual)")
    plt.xlim(0,100)
    plt.legend()
    plt.grid(True)
    plt.close()

    # ======================================
    # Análisis espectral (tabla) sobre residual
    # ======================================
    analisis = []
    for canal_label in sorted(original_to_col.keys()):
        col = original_to_col[canal_label]
        señal = matriz_eeg[:, col]
        frecuencias_psd, potencia_psd = welch(señal, fs=Fs, nperseg=4096)
        indice = np.argmax(potencia_psd)
        frecuencia_pico = frecuencias_psd[indice]
        potencia_pico = potencia_psd[indice]

        # Buscar región (por índice original 0-based)
        region = ""
        for nombre, canales_range in regiones_original.items():
            if (canal_label - 1) in canales_range:
                region = nombre
                break

        analisis.append([canal_label, region, frecuencia_pico, potencia_pico])

    df_espectral = pd.DataFrame(analisis, columns=["Canal", "Región", "Frecuencia_Pico_Hz", "Potencia_Pico"]) 
    df_espectral = df_espectral.round(2)
    print(df_espectral)
    df_espectral.to_csv("Analisis_Espectral.csv", index=False, encoding="utf-8-sig")
    print("\nArchivo guardado: Analisis_Espectral.csv")

    
    # ======================================
    # GRAFICO 1 PCA de los estadísticos descriptivos
    # ======================================

    print("\nRealizando PCA...")

    # Seleccionar solamente las variables numéricas
    X = df.drop(columns=["Canal", "Región"])
    df = df.round(1)

    # Estandarizar los datos
    scaler = StandardScaler()
    X_escalado = scaler.fit_transform(X)

    # Ejecutar PCA
    pca = PCA()
    componentes = pca.fit_transform(X_escalado)
    print(componentes.shape)
    print(componentes[:5, 2])
    print("\nVarianza explicada")

    for i, porcentaje in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {porcentaje*100:.2f}%")

    # ======================================
    # Guardar varianza explicada
    # ======================================

    varianza = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
        "Varianza (%)": pca.explained_variance_ratio_ * 100
    })

    varianza.to_csv(
        "PCA_varianza.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ======================================
    # Scree Plot GRAFICA 1 
    # ======================================
    plt.figure(figsize=(8,5))

    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_ * 100,
        marker="o"
    )

    plt.xlabel("Componente principal")
    plt.ylabel("Varianza explicada (%)")
    plt.title("Scree Plot")

    plt.grid(True)

    plt.tight_layout()
    plt.close()

    df_pca = pd.DataFrame({
    "Canal": df["Canal"],
    "Región": df["Región"],
    "PC1": componentes[:,0],
    "PC2": componentes[:,1],
    "PC3": componentes[:,2]
    })

    print("Entrando al gráfico 3D...")

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection="3d")

    colores = {
        "CPF":"red",
        "Nacc":"blue",
        "Amy":"green",
        "Hyp":"purple"
    }

    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        ax.scatter(
            datos_region["PC1"],
            datos_region["PC2"],
            datos_region["PC3"],
            color=colores[region],
            label=region,
            s=80
        )
        # Etiqueta de cada canal
        for _, fila in datos_region.iterrows():
            ax.text(
                fila["PC1"] + 0.03,
                fila["PC2"] + 0.03,
                fila["PC3"] + 0.03,
                str(int(fila["Canal"])),
                fontsize=9,
                color=colores[region]
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA tridimensional")
    ax.legend()
    plt.tight_layout()
    plt.close()
    print("Gráfico 3D finalizado.")

    # ======================================
    # BIPLOT PCA  GRAFICO 3
    # ======================================

    plt.figure(figsize=(10,8))

    # Dibujar los puntos
    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        plt.scatter(
            datos_region["PC1"],
            datos_region["PC2"],
            color=colores[region],
            s=120,
            alpha=0.4
        )
        for _, fila in datos_region.iterrows():
            plt.text(
                fila["PC1"] + 0.03,
                fila["PC2"] + 0.03,
                str(int(fila["Canal"])),
                fontsize=10,
                fontweight="bold",
                color=colores[region],
                ha="center",
                va="center"
            )

    # Cargas (loadings)
    loadings = pca.components_.T

    for i, variable in enumerate(X.columns):
        plt.arrow(
            0,
            0,
            loadings[i,0]*4,
            loadings[i,1]*4,
            color="black",
            alpha=0.7,
            head_width=0.05
        )
        plt.text(
            loadings[i,0]*4.2,
            loadings[i,1]*4.2,
            variable,
            fontsize=9
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot del PCA")
    plt.axhline(0,color="gray",linewidth=0.5)
    plt.axvline(0,color="gray",linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.close()

    df_pca.to_csv(
        "PCA_canales.csv",
        index=False,
        encoding="utf-8-sig"
    )
    
    print("\nArchivo PCA guardado correctamente.")
    # GRAFUICO 4
    plt.figure(figsize=(8,6))

    colores = {
        "CPF":"red",
        "Nacc":"blue",
        "Amy":"green",
        "Hyp":"purple"
    }

    for region in colores:
        datos_region = df_pca[df_pca["Región"] == region]
        plt.scatter(
            datos_region["PC1"],
            datos_region["PC2"],
            color=colores[region],
            s=120,
            alpha=0.5,
            label=region
        )
        for _, fila in datos_region.iterrows():
            plt.text(
                fila["PC1"],
                fila["PC2"],
                str(int(fila["Canal"])),
                fontsize=10,
                fontweight="bold",
                color=colores[region],
                ha="center",
                va="center"
            )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA de los canales")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.close()


    print("\nDetección de artefactos")
    print("-----------------------")

    # Detección de artefactos usando etiquetas originales
    display_channels = sorted(original_to_col.keys())
    for label in display_channels:
        col = original_to_col[label]
        señal = matriz_eeg[:, col]
        diff = np.diff(señal)
        umbral = 5 * np.std(diff)
        artefactos = np.where(np.abs(diff) > umbral)[0]
        print(f"Canal {label}: {len(artefactos)} posibles artefactos")

    # Visualización de todos los canales
    segundos = 5
    muestras = int(segundos * Fs)
    tiempo = np.arange(muestras) / Fs
    plt.figure(figsize=(18,12))

    offset = 300

    # Definición de regiones y colores
    regiones= {
        "CPF": ("red", range(0,10)),
        "Nacc": ("blue", range(10,16)),
        "Amy": ("green", range(16,25)),
        "Hyp": ("purple", range(25,32))
    }

    # Mostrar todos los canales remanentes (etiquetas originales)
    display_channels = sorted(original_to_col.keys())
    for idx, label in enumerate(display_channels):
        # determinar región y color
        color = "black"
        for nombre_region, (col_color, canales_range) in regiones.items():
            if (label - 1) in canales_range:
                color = col_color
                break
        col = original_to_col[label]
        plt.plot(
            tiempo,
            matriz_eeg[:muestras, col] + idx*offset,
            color=color,
            linewidth=0.8
        )

    plt.yticks(
        np.arange(len(display_channels))*offset,
        [f"{lbl}" for lbl in display_channels]
    )

    plt.xlabel("Tiempo (s)")
    plt.ylabel("Canales")
    plt.title("Actividad neuronal por regiones cerebrales")

    plt.grid(True)

    # Leyenda
    from matplotlib.lines import Line2D

    leyenda = [
        Line2D([0],[0],color="red",label="CPF"),
        Line2D([0],[0],color="blue",label="Nacc"),
        Line2D([0],[0],color="green",label="Amy"),
        Line2D([0],[0],color="purple",label="Hyp")
    ]

    plt.legend(handles=leyenda)

    plt.tight_layout()
    plt.savefig("allChan_activity.png", dpi=300, bbox_inches="tight")
    plt.close()