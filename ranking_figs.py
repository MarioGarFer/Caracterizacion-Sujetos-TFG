import matplotlib.pyplot as plt
import pandas as pd
import os

"""
Genera gráficos de detección de frecuencias por sujeto y ensayo en pdf
"""

# Cargar el archivo combinado y el de clasificación
df = pd.read_csv("./results/fft_detection_results_combined.csv")
classification_df = pd.read_csv("./results/sorted_detection_rankings.csv")

# Crear carpeta base para guardar gráficos
base_output_dir = "./plots/"
os.makedirs(base_output_dir, exist_ok=True)

# Recorrer cada sujeto
for subject in df["Subject"].unique():
    subject_data = df[df["Subject"] == subject]
    classification_row = classification_df[classification_df["Subject"] == subject]
    if classification_row.empty:
        continue
    group = classification_row["Category"].values[0].lower()

    # Crear carpeta del grupo
    output_dir = f"{base_output_dir}/{group}"
    os.makedirs(output_dir, exist_ok=True)

    # Crear figura con 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 6))
    axs = axs.flatten()  # Convertimos en lista para indexar fácilmente

    for trial in range(4):
        trial_data = subject_data[subject_data["Trial"] == trial]
        if trial_data.empty:
            continue

        detected_time_percentage = trial_data.apply(
            lambda row: (
                10
                if row["SignalDetected"] == 1 and row["DetectedTimePercentage_5dB"] == 0
                else row["DetectedTimePercentage_5dB"] if row["SignalDetected"] == 1 else 0
            ),
            axis=1,
        )

        axs[trial].plot(
            trial_data["Freq"], detected_time_percentage, marker="", color=["blue", "red", "orange", "green"][trial]
        )
        axs[trial].set_title(f"Trial {trial + 1}")
        axs[trial].set_xlabel("Frecuencia (Hz)")
        axs[trial].set_ylabel("Porcentaje de Detección")
        axs[trial].set_ylim(-5, 105)

    plt.suptitle(f"Porcentaje de Frecuencias por Ensayo - Sujeto {subject} ({group.title()})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Guardar gráfico
    output_path = f"{output_dir}/plot_sujeto_{subject}_{group}.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

print("Gráficos generados correctamente por grupo y sujeto.")
