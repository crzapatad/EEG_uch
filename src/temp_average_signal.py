import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nombre_archivo = 'allChan_1kHz_clean.mat'
with h5py.File(nombre_archivo, 'r') as datos:
    matriz_eeg = np.array(datos['allChan_clean'])
    Fs = np.array(datos['Fs'])[0][0]

n_muestras, n_canales = matriz_eeg.shape

# Definición de regiones y canales (0-based)
regiones = {
    'CPF': list(range(0,10)),
    'Nacc': list(range(10,16)),
    'Amy': list(range(16,25)),
    'Hyp': list(range(25,32))
}

# Excluir canales 29 y 30 (1-based -> 29,30). 0-based indices: 28,29
excluir = {28,29}

resultados = []
fig, ax = plt.subplots(figsize=(10,6))
for nombre, canales in regiones.items():
    canales_filtrados = [c for c in canales if c not in excluir]
    if len(canales_filtrados) == 0:
        continue
    señal_promedio = np.mean(matriz_eeg[:, canales_filtrados], axis=1)
    t = np.arange(len(señal_promedio)) / Fs
    resultados.append((nombre, canales_filtrados, señal_promedio))
    ax.plot(t[:int(Fs*5)], señal_promedio[:int(Fs*5)], label=f"{nombre} (N={len(canales_filtrados)})")

ax.set_xlabel('Tiempo (s)')
ax.set_ylabel('Amplitud')
ax.set_title('Señal promedio por región (excluyendo canales 29 y 30)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('SeñalPromedio_regiones_excluyendo_29_30.png', dpi=300)

# Guardar CSV con promedio por región (toma las primeras 10k muestras para tamaño manejable)
max_muestras = min(10000, n_muestras)
df = pd.DataFrame({ 'Tiempo_s': np.arange(max_muestras)/Fs })
for nombre, canales, señal in resultados:
    df[f'{nombre}_avg'] = señal[:max_muestras]

df.to_csv('SeñalPromedio_regiones_excluyendo_29_30.csv', index=False, encoding='utf-8-sig')
print('CSV y PNG generados')
