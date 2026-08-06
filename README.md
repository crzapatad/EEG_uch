# Análisis y Procesamiento de Señales Neurofisiológicas

Este proyecto contiene un pipeline en Python para la **carga, procesamiento y análisis de señales neurofisiológicas multicanal**, principalmente EEG/LFP.

## Objetivo

El programa permite procesar registros obtenidos desde archivos `.mat` y datos provenientes de **Open Ephys**, realizando diferentes análisis temporales, estadísticos, espectrales y de fase para caracterizar la actividad neuronal y estudiar relaciones entre canales y regiones cerebrales.

## Flujo de procesamiento

```text
Datos neurofisiológicos
        ↓
Carga de señales
        ↓
Submuestreo opcional
        ↓
Estadística descriptiva
        ↓
Promedios regionales
        ↓
Señal residual
        ↓
┌───────────────┬──────────────┬──────────────┐
│ Análisis      │ Análisis     │ PCA          │
│ espectral     │ de fase      │              │
│               │              │              │
│ PSD           │ Hilbert      │ Scree Plot   │
│ Espectrograma │ MRL          │ Biplot       │
│ Frecuencia    │ Desfase      │ PCA 2D / 3D  │
│ dominante     │ temporal     │              │
└───────────────┴──────────────┴──────────────┘
        ↓
Resultados CSV / NPY / MAT / PNG
```

## Procesamiento realizado

### 1. Carga de datos

* Lectura de archivos `.mat`.
* Lectura de registros `.continuous` de Open Ephys.
* Identificación de frecuencia de muestreo y canales.
* Organización de las señales en una matriz multicanal.

### 2. Submuestreo

Permite reducir la frecuencia de muestreo de las señales para disminuir el volumen de datos y facilitar su procesamiento.

### 3. Estadística descriptiva

Se calculan características por canal, entre ellas:

* Promedio.
* Desviación estándar.
* Mediana.
* MAD.
* Mínimo y máximo.
* Rango.
* Estadísticas de la primera diferencia.

### 4. Análisis regional y señal residual

Los canales se agrupan en regiones cerebrales:

* **CPF:** canales 1–10
* **Nacc:** canales 11–16
* **Amy:** canales 17–25
* **Hyp:** canales 26–32

Se calcula el promedio de actividad de cada región y posteriormente una señal residual:

```text
Señal residual = Señal del canal − Promedio de su región
```

Esto permite estudiar la actividad específica de cada canal respecto de la actividad promedio regional.

### 5. Análisis espectral

Se utiliza el método de Welch para calcular el **Power Spectral Density (PSD)** y caracterizar la distribución de potencia según la frecuencia.

También se identifican:

* Frecuencia dominante.
* Potencia máxima.
* Banda de frecuencia correspondiente.

Además, se generan **espectrogramas** para observar cómo cambia la potencia de las distintas frecuencias a través del tiempo.

### 6. Análisis de fase

Se utiliza la transformada de Hilbert para obtener la **fase instantánea** de las señales.

También se analiza la diferencia de fase entre canales, incluyendo:

* Diferencia de fase.
* Coherencia mediante **Mean Resultant Length (MRL)**.
* Relación entre fase, frecuencia y desfase temporal.

### 7. Análisis PCA

Se aplica **Análisis de Componentes Principales (PCA)** sobre las características estadísticas de los canales.

El PCA permite reducir la dimensionalidad y visualizar similitudes y diferencias entre los 32 canales mediante:

* Scree Plot.
* PCA 2D.
* PCA 3D.
* Biplot.
* Loadings y varianza explicada.

### 8. Detección de posibles artefactos

Se analizan cambios bruscos en las señales mediante la primera diferencia y un umbral basado en la desviación estándar para identificar posibles artefactos.

## Resultados

El pipeline puede generar:

* Archivos `.CSV` con estadísticas y resultados numéricos.
* Archivos `.NPY` y `.MAT` con señales procesadas.
* Gráficos `.PNG` de PSD y espectrogramas.
* Visualizaciones PCA 2D, 3D, biplot y scree plot.
* Resultados de análisis de fase y coherencia.

## Tecnologías utilizadas

* Python
* NumPy
* Pandas
* SciPy
* Matplotlib
* Scikit-learn
* h5py
* Open Ephys

## Resumen

Este proyecto implementa un pipeline para transformar **registros neurofisiológicos multicanal en información cuantitativa y visualizable**, combinando procesamiento de señales, análisis estadístico, análisis espectral, análisis de fase y reducción de dimensionalidad mediante PCA.

