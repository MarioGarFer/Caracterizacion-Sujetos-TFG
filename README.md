# Caracterización de sujetos en interfaces cerebro-máquina mediante el análisis de la evolución temporal de potenciales visuales evocados

Este repositorio contiene el código desarrollado para un Trabajo de Fin de Grado (TFG) centrado en el análisis de señales EEG obtenidas mediante estimulación visual periódica (SSVEP).  
El objetivo principal del proyecto es estudiar la **variabilidad intersujeto** en estas señales y analizar hasta qué punto es posible **caracterizar y clasificar a los usuarios** según métricas extraídas de su rendimiento.

## Objetivos del proyecto

- Analizar cómo responden distintos sujetos a estímulos visuales en un sistema BCI basado en SSVEP.
- Estudiar la estabilidad temporal de la detección de frecuencias a lo largo del experimento.
- Extraer características cuantificables que permitan clasificar a los sujetos según su rendimiento (por ejemplo, Good / Mid / Bad).
- Evaluar la utilidad de cada característica para predecir métricas de rendimiento como el ITR o la precisión binaria (BCI performance).

## Estructura del repositorio

- `extracion_features.py`  
  Script principal que extrae las características básicas por sujeto a partir de los datos crudos.

- `new_features.ipynb`  
  Notebook para el desarrollo y prueba de nuevas características más avanzadas, y su análisis exploratorio.

- `ranking.py`  
Archivo para calcular el ranking de sujetos según las métricas principales (SignalDetected y DetectedTimePercentage), y clasificarlos en categorías (Good, Mid, Bad).

- `ranking_figs.py`  
  Script para generar representaciones gráficas del ranking de sujetos y visualizaciones por grupo.



## Requisitos

Este proyecto está escrito en **Python 3**. Se recomienda crear un entorno virtual con las siguientes librerías:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `seaborn`
- `jupyter` (para los notebooks)

Puedes instalar los requisitos básicos con:

```bash
pip install -r requirements.txt
