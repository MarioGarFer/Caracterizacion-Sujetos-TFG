import pandas as pd

"""
Este script lee un archivo CSV que contiene resultados de detección de FFT y clasifica a los sujetos en categorías basadas en el número de detecciones.
Las categorías son "Good", "Mid" y "Bad", dependiendo de la cantidad de detecciones realizadas para cada sujeto.
"""

# Leer el archivo CSV
file_path = "./results/fft_detection_results_combined.csv"
df = pd.read_csv(file_path)

# Filtrar las filas donde 'Detected' es 1
detected_df = df[df["SignalDetected"] == 1]

# Contar el número de detecciones por cada sujeto
detection_count = detected_df.groupby("Subject").size().reset_index(name="DetectionCount")

# Ordenar por el número de detecciones (de mayor a menor)
sorted_detection_count = detection_count.sort_values(by="DetectionCount", ascending=False)

# Clasificación basada en el ranking
sorted_detection_count["Rank"] = sorted_detection_count["DetectionCount"].rank(ascending=False)


# Definir las categorías de clasificación
def classify(rank, total):
    if rank <= 22:
        return "Good"
    elif rank > total - 22:
        return "Bad"
    else:
        return "Mid"


# Aplicar la clasificación a cada sujeto
total_subjects = len(sorted_detection_count)
sorted_detection_count["Category"] = sorted_detection_count["Rank"].apply(lambda x: classify(x, total_subjects))

# Guardar el resultado en un nuevo archivo CSV
sorted_file_with_category = "./results/sorted_detection_rankings.csv"
sorted_detection_count.to_csv(sorted_file_with_category, index=False)

sorted_file_with_category
