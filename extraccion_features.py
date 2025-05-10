import numpy as np
import scipy.io
import scipy.signal as signal
import matplotlib.pyplot as plt

"""
Calcula todas las métricas de potencia y detección para cada sujeto, cada frecuencia y cada intento.
Guarda los resultados en un archivo CSV
"""

# Archivo CSV único
csv_filename = "./results/fft_detection_results_combined.csv"
with open(csv_filename, "w") as file:
    file.write(
        "Subject,Trial,Freq,SignalDetected,AvgPower,PowerVariability,DetectedTimePercentage_1dB,DetectionTime_1dB,DetectedTimePercentage_3dB,DetectionTime_3dB,DetectedTimePercentage_5dB,DetectionTime_5dB\n"
    )


# Función para guardar los resultados en CSV
def save_to_csv(data):
    """
    Guarda los resultados en un archivo CSV.
    Parámetros:
        - data: Diccionario con los resultados a guardar.
    """

    with open(csv_filename, "a") as file:
        file.write(
            f"{data['subject']},{data['trial']},{data['freq']},{data['detected']},"
            f"{data['avg_power']:.2f},{data['power_variability']:.2f},{data['detected_time_percentage_1dB']:.2f},{data['detection_time_1dB']:.2f},"
            f"{data['detected_time_percentage_3dB']:.2f},{data['detection_time_3dB']:.2f},"
            f"{data['detected_time_percentage_5dB']:.2f},{data['detection_time_5dB']:.2f}\n"
        )


# Función para calcular el tiempo hasta la detección
def calculate_detection_time(intensity_at_target_freq_db, intensity_threshold):
    """
    Calcula el tiempo hasta la detección de la señal.
    Parámetros:
        - intensity_at_target_freq_db: Intensidad de la señal en dB.
        - intensity_threshold: Umbral de intensidad en dB/Hz.
    Retorna:
        - El primer instante donde la intensidad supera el umbral.
        - np.nan si no se detecta la señal.
    """

    # Buscar el primer instante donde la intensidad supera el umbral
    above_threshold_indices = np.where(intensity_at_target_freq_db > intensity_threshold)[0]
    if len(above_threshold_indices) > 0:
        return above_threshold_indices[0]  # Primer instante donde se detecta la señal
    return np.nan  # No detectado


# Función para aplicar FFT y detectar la señal
def detect_signal_fft(eeg_data, sampling_rate, target_freq):
    """
    Aplica FFT a la señal EEG y detecta si la frecuencia esperada está presente
    con un margen de tolerancia de 0.5 Hz.

    Parámetros:
        - eeg_data: Señal EEG (1D numpy array).
        - sampling_rate: Frecuencia de muestreo en Hz.
        - target_freq: Frecuencia esperada a detectar.

    Retorna:
        - detected: 1 si la señal fue detectada, 0 si no.
    """

    # Aplicar FFT
    fft_result = np.fft.fft(eeg_data)
    freqs = np.fft.fftfreq(len(eeg_data), 1 / sampling_rate)
    magnitude = np.abs(fft_result) / len(eeg_data)

    # Filtrar frecuencias positivas
    positive_freqs = freqs[freqs >= 0]
    positive_magnitude = magnitude[freqs >= 0]

    freq_tolerance = 0.5

    # Detected a 1 si la frecuencia esperada es el pico máximo y se encuentra dentro de la tolerancia
    peak_idx = np.argmax(positive_magnitude)
    peak_freq = positive_freqs[peak_idx]
    detected = 1 if abs(peak_freq - target_freq) <= freq_tolerance else 0

    return detected


# Función para calcular métricas de potencia
def calculate_avg_power(eeg_data, sampling_rate, freq_idx, stimulus_frequencies, intensity_threshold):
    """
    Calcula la potencia promedio, el porcentaje de tiempo detectado, el tiempo hasta la detección y la
    variabilidad de la potencia para una señal EEG dada.

    Parámetros:
        - eeg_data: Señal EEG (1D numpy array).
        - sampling_rate: Frecuencia de muestreo en Hz.
        - freq_idx: Índice de la frecuencia objetivo en el array de frecuencias.
        - stimulus_frequencies: Array de frecuencias del sujeto.
        - intensity_threshold: Umbral de intensidad en dB/Hz.
    Retorna:
        - avg_power: Potencia promedio en dB.
        - detected_time_percentage: Porcentaje de tiempo detectado.
        - detection_time: Tiempo hasta la detección.
        - power_variability: Variabilidad de la potencia.
    """

    # Filtrar la señal EEG para centrarse en la banda de frecuencias de interés
    low_freq, high_freq = 5, 40
    b, a = signal.butter(4, [low_freq, high_freq], fs=sampling_rate, btype="band")
    filtered_data = signal.lfilter(b, a, eeg_data)

    # Cálculo del espectrograma
    Pxx, freqs, bins, _ = plt.specgram(filtered_data, NFFT=300, Fs=sampling_rate, noverlap=int(300 / 4), cmap="jet")

    stimulus_freq = stimulus_frequencies[freq_idx]
    freq_tolerance = 0.5

    # Filtrar las frecuencias para centrarse en la frecuencia objetivo
    freq_indices = np.where((freqs >= stimulus_freq - freq_tolerance) & (freqs <= stimulus_freq + freq_tolerance))[0]

    # Calcular la intensidad promedio en la frecuencia objetivo
    intensity_at_target_freq = np.mean(Pxx[freq_indices, :], axis=0)
    intensity_at_target_freq_db = 10 * np.log10(intensity_at_target_freq + 1e-10)

    # Calcular la característica de potencia y detección
    avg_power = np.mean(intensity_at_target_freq_db)
    detected_time_percentage = np.mean(intensity_at_target_freq_db > intensity_threshold) * 100
    detection_time = calculate_detection_time(intensity_at_target_freq_db, intensity_threshold)
    power_variability = np.std(intensity_at_target_freq_db)

    return avg_power, detected_time_percentage, detection_time, power_variability


# Función principal de ejecución
def execute(subject, condition_freq, trial):
    """
    Ejecuta el proceso de detección de señales y cálculo de métricas para un sujeto, frecuencia y ensayo dados.
    Parámetros:
        - subject: número del sujeto.
        - condition_freq: Frecuencia objetivo a detectar.
        - trial: Número de ensayo.
    """

    dataset_path = "./datasets"
    channel_idx = 61  # Canal Oz
    file_name = f"{dataset_path}/S{subject}.mat"
    data = scipy.io.loadmat(file_name)

    # Cargar los datos EEG del sujeto
    sampling_rate = 250
    time_without_stimuli = 0.5
    x = int(time_without_stimuli * sampling_rate)

    eeg_data = data["data"][0, 0]["EEG"]
    eeg_data_stimulus_only = eeg_data[:, x:-x, :, :]

    # Obtener el array de frecuencias del sujeto
    frequencies = data["data"][0, 0]["suppl_info"]["freqs"][0, 0][0]

    # Buscar el índice de la frecuencia más cercana a condition_freq
    condition_idx = (np.abs(frequencies - condition_freq)).argmin()

    # (Opcional) Comprobación: si la diferencia es muy grande, saltar esa frecuencia
    if abs(frequencies[condition_idx] - condition_freq) > 0.05:
        print(
            f"⚠️ Frecuencia {condition_freq} no encontrada exactamente en sujeto {subject}, se usará {frequencies[condition_idx]}"
        )

    # Obtener la señal EEG del canal Oz para el sujeto, ensayo y frecuencia específicos
    eeg_signal = eeg_data_stimulus_only[channel_idx, :, trial, condition_idx]

    # Detectar la señal usando FFT
    detected = detect_signal_fft(eeg_signal, sampling_rate, condition_freq)
    stimulus_frequencies = data["data"][0, 0]["suppl_info"]["freqs"][0, 0][0]

    # Para cada threshold, calcular las métricas
    thresholds = [1, 3, 5]
    results = {"subject": subject, "trial": trial, "freq": condition_freq, "detected": detected}

    for t in thresholds:
        avg_power, detected_time_percentage, detection_time, power_variability = calculate_avg_power(
            eeg_signal, sampling_rate, condition_idx, stimulus_frequencies, intensity_threshold=t
        )
        results[f"avg_power"] = avg_power
        results[f"detected_time_percentage_{t}dB"] = detected_time_percentage
        results[f"detection_time_{t}dB"] = detection_time
        results[f"power_variability"] = power_variability

    # Guardar los resultados en CSV
    save_to_csv(results)


if __name__ == "__main__":
    subjects = list(range(1, 71))
    trial = list(range(4))

    # Generar las frecuencias desde 8 hasta 15.6 con un paso de 0.2
    freqs = np.arange(8, 16, 0.2)  # np.arange no incluye el límite superior, por eso usamos 15.8

    # Redondear a un decimal cada frecuencia (por si hay problemas de precisión)
    list_freqs = [round(freq, 1) for freq in freqs]

    # Eliminar duplicados (por si el redondeo produce alguno) y ordenar la lista
    list_freqs = sorted(set(list_freqs))

    for subject in subjects:
        print(f"Processing subject {subject}")
        for freq in list_freqs:
            for t in trial:
                execute(subject, freq, t)
