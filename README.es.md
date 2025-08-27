# HydroMassNet — Redes Neuronales Bayesianas para Masa de Hidrógeno Neutro

[Português](README.pt.md) | [English](README.en.md) | [Español](README.es.md)

HydroMassNet es un pipeline de extremo a extremo para estimar la masa de hidrógeno neutro en galaxias (log10(M_HI)) usando Redes Neuronales Bayesianas (BNNs). Cubre adquisición de datos (catálogos astronómicos), preprocesamiento, ingeniería de características, entrenamiento/evaluación con incertidumbre, comparación de modelos y generación de figuras para publicación.

- Stack principal: TensorFlow 2.x, TensorFlow Probability, scikit-learn, LightGBM
- Acceso a datos astronómicos: astroquery (VizieR, XMatch)
- Configuración reproducible vía YAML

## Estructura del repositorio

- data/: datasets crudos y procesados
- results/
  - saved_models/: pesos entrenados y artefactos (incluye CSVs de predicciones)
  - plots/: figuras generadas durante la evaluación
  - search_results/: reportes de selección de características y búsqueda de hiperparámetros
- config/
  - config.yaml: configuraciones del proyecto (rutas, datos, entrenamiento, selección de características, campeones)
- src/
  - download_dataset.py: construye el catálogo completo (VizieR + cross-match opcional)
  - preprocess.py: limpieza, imputación, ingeniería de características y guardado del CSV procesado
  - train.py: entrenamiento genérico (bayesianos y vanilla) para experimentos
  - evaluate.py: evalúa un modelo entrenado y guarda predicciones con incertidumbre
  - predict.py: predicción de una muestra con incertidumbre
  - baseline.py: baseline de regresión lineal
  - feature_selection.py: exploración/selección de características con LightGBM
  - run_full_optimization.py: etapa 1 (selección) + etapa 2 (búsqueda de hiperparámetros)
  - plotting.py, publication_plots.py: utilidades de gráficos
  - models/: capas y arquitecturas Bayesianas
  - utils/: utilidades auxiliares
- run_pipeline.py: orquesta preprocess -> train -> evaluate -> baseline -> plots
- run_champions.py: entrena/evalúa los modelos campeones definidos en el config
- pyproject.toml / requirements.txt: dependencias
- LICENSE: MIT

## Requisitos

- Python >= 3.9
- Poetry (recomendado) o pip/venv
- Opcional: GPU con TensorFlow configurado

## Instalación

- Poetry (recomendado)
  - git clone https://github.com/joelsonsartori/HydroMassNet.git
  - cd HydroMassNet
  - poetry install

- pip
  - git clone https://github.com/joelsonsartori/HydroMassNet.git
  - cd HydroMassNet
  - python -m venv venv y source venv/bin/activate  # Windows: venv\Scripts\activate
  - pip install -r requirements.txt

## Configuración

La configuración principal está en config/config.yaml (algunos scripts también buscan ./config.yaml cuando se ejecutan en la raíz). Secciones clave:

- paths: raw_data, processed_data, saved_models, plots, search_results
- data_processing: divisiones de validación/test y la lista de características del dataset procesado
- training: epochs, patience
- feature_selection: características candidatas, min_features, top_n_to_tune
- champion_models: características por modelo e hiperparámetros finales (bnn, dbnn, vanilla)

## Inicio rápido

1) Construir el catálogo completo (descarga/merge vía astroquery)
- poetry run python src/download_dataset.py

2) Preprocesar e ingenierizar características
- poetry run python src/preprocess.py

3) Entrenar + evaluar todo y generar gráficos (orquestación)
- poetry run python run_pipeline.py

Esto ejecuta bnn, dbnn y vanilla, produce CSVs de predicciones en results/saved_models y figuras en results/plots.

## Uso manual

- Entrenar un modelo específico
  - poetry run python src/train.py --model bnn
  - poetry run python src/train.py --model dbnn
  - poetry run python src/train.py --model vanilla
  - Flags opcionales:
    - --learning_rate, --batch_size
    - --hidden_layers para bnn/vanilla (ej.: 256-128)
    - --core_layers/--head_layers para dbnn (ej.: 512-256 y 128-64)
    - --dropout para vanilla
    - --features "iMAG,e_iMAG,logMsT,..." para sobrescribir las características

- Evaluar un modelo entrenado (genera {model}_predictions.csv)
  - poetry run python src/evaluate.py --model bnn|dbnn|vanilla

- Baseline
  - poetry run python src/baseline.py

- Gráficos
  - Generados automáticamente por run_pipeline.py (históricos, comparación de predicciones e intervalos de confianza)

- Predecir una muestra con incertidumbre
  - poetry run python src/predict.py --model bnn --input_values <v1 v2 ... vN> --config config/config.yaml
  - Nota: input_values debe igualar la longitud de data_processing.features; usa scaler_x.pkl y scaler_y.pkl.

- Optimización completa (selección de características + búsqueda de hiperparámetros)
  - poetry run python src/run_full_optimization.py

- Modelos campeones (entrenamiento/evaluación final según config.champion_models)
  - poetry run python run_champions.py

## Datos y características

- Objetivo por defecto: logMHI
- Ejemplo de características (del config): iMAG, e_iMAG, logMsT, logSFR22, e_logMsT, Dist, RVel, Ag, Ai, surface_brightness_proxy
- El proxy surface_brightness_proxy se deriva en el preprocesamiento a partir de iMAG y la razón de ejes b/a
- Datos guardados en CSVs bajo data/ según configuración

## Reproducibilidad

- Semilla global en YAML (seed: 1601)
- Pipelines basados en configuración y preprocesamiento determinista cuando es posible

## Resultados y artefactos

- results/saved_models/: pesos (*.weights.h5), CSVs de predicciones, históricos
- results/plots/: figuras PNG/PDF de métricas de entrenamiento y visualización de incertidumbre
- results/search_results/: reportes CSV de selección de características y optimización

## Licencia

Proyecto bajo licencia MIT. Ver LICENSE para más detalles.

## Citación

Si usas HydroMassNet en trabajos académicos, por favor cita este repositorio. Se añadirá un BibTeX cuando haya preprint.

## Agradecimientos

- Servicios VizieR y XMatch vía astroquery
- TensorFlow Probability para capas Bayesianas
- Librerías de la comunidad: scikit-learn, LightGBM, pandas, matplotlib/seaborn
