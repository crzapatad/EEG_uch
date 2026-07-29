import matplotlib
matplotlib.use("TkAgg")
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Nombre del archivo
nombre_archivo = "allChan_1kHz_clean.mat"

# Abrir archivo .mat
with h5py.File(nombre_archivo, "r") as datos:

    # Mostrar variables disponibles
    print("\nVariables disponibles:")
    print("----------------------")
    for variable in datos.keys():
        print(variable)

    print("\nContenido del archivo:")
    print("----------------------")
    def recorrer(nombre, objeto):
        print(nombre)

    datos.visititems(recorrer)

    # ======================================
    # Mostrar configuración de canales
    # ======================================

    print("\nConfiguración de canales")
    print("------------------------")

    config = datos["processing"]["channel_configuration"]

    for region in config.keys():

        contenido = np.array(config[region])
        print(f"\nRegión: {region}")
        print("Forma:", contenido.shape)
        print(contenido)
    # Leer matriz principal
    matriz_eeg = np.array(datos["allChan_clean"])

    # Leer frecuencia de muestreo
    Fs = np.array(datos["Fs"])[0][0]

    print("\nInformación del registro")
    print("------------------------")
    print(f"Frecuencia de muestreo: {Fs} Hz")
    print(f"Dimensiones: {matriz_eeg.shape}")

    n_muestras = matriz_eeg.shape[0]
    n_canales = matriz_eeg.shape[1]

    duracion = n_muestras / Fs

    print(f"Número de canales: {n_canales}")
    print(f"Número de muestras: {n_muestras}")
    print(f"Duración: {duracion:.2f} segundos")
    print(f"Duración: {duracion/60:.2f} minutos")

    # ======================================
    # Estadísticos descriptivos por canal
    # ======================================

    estadisticas = []

    # Diccionario región -> canales
    regiones = {
        "CPF": range(0, 10),
        "Nacc": range(10, 16),
        "Amy": range(16, 25),
        "Hyp": range(25, 32)
    }

    for canal in range(n_canales):
        señal = matriz_eeg[:, canal]
        # Buscar a qué región pertenece
        region = ""
        for nombre, canales in regiones.items():
            if canal in canales:
                region = nombre
                break

        promedio = np.mean(señal)
        desviacion = np.std(señal)
        mediana = np.median(señal)

        # Median Absolute Deviation (MAD)
        mad = np.median(np.abs(señal - mediana))
        minimo = np.min(señal)
        maximo = np.max(señal)
        rango = maximo - minimo

        # Estadísticos del diff
        diff_promedio = np.mean(np.diff(señal))
        diff = np.diff(señal)
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

    # Crear DataFrame
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

    # Guardar resultados
    df.to_csv(
    "estadisticas_canales.csv",
    index=False,
    encoding="utf-8-sig"
    )

    print("\nArchivo guardado como:")
    print("estadisticas_canales.csv")

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
    # Loadings del PCA
    # ======================================

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(len(X.columns))]
    )

    print("\nLoadings del PCA")
    print(loadings.round(3))

    loadings.to_csv(
        "PCA_loadings.csv",
        encoding="utf-8-sig"
    )

    print("\nArchivo guardado como:")
    print("PCA_loadings.csv")

    # ======================================
    # Contribución (%) de cada variable
    # ======================================

    # Cuadrado de los loadings
    contribuciones = loadings**2

    # Convertir cada columna en porcentaje
    contribuciones = contribuciones.div(
        contribuciones.sum(axis=0),
        axis=1
    ) * 100

    print("\nContribución porcentual de cada variable")
    print(contribuciones.round(2))

    contribuciones.to_csv(
        "PCA_contribuciones.csv",
        encoding="utf-8-sig"
    )

    print("\nArchivo guardado como:")
    print("PCA_contribuciones.csv")

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
    plt.show()

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

    plt.show()
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
    plt.show()

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
    plt.show()


    print("\nDetección de artefactos")
    print("-----------------------")

    for canal in range(n_canales):
        señal = matriz_eeg[:, canal]
        # Primera diferencia
        diff = np.diff(señal)
        # Umbral para detectar cambios bruscos
        umbral = 5 * np.std(diff)
        # Índices donde se detectan posibles artefactos
        artefactos = np.where(np.abs(diff) > umbral)[0]
        print(
            f"Canal {canal+1}: "
            f"{len(artefactos)} posibles artefactos"
        )

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

    for nombre_region, (color, canales) in regiones.items():
        for canal in canales:
            plt.plot(
                tiempo,
                matriz_eeg[:muestras, canal] + canal*offset,
                color=color,
                linewidth=0.8
            )

    plt.yticks(
        np.arange(n_canales)*offset,
        [f"{i+1}" for i in range(n_canales)]
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
    plt.show()