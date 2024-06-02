## Español

[![English](https://img.shields.io/badge/lang-English-blue)](README.md)
[![Português](https://img.shields.io/badge/lang-Português-green)](README.pt.md)

# HydroMassNet

HydroMassNet es un proyecto destinado a crear redes neuronales Bayesianas para identificar el porcentaje de masa de hidrógeno neutro en galaxias.

### Requisitos Previos

- Python 3.8 o superior
- Poetry
- TensorFlow
- TensorFlow Probability
- Scikit-learn
- Matplotlib
- Seaborn

### Instalación

Puede instalar las dependencias usando Poetry o pip.

#### Usando Poetry

1. Clone el repositorio:
    ```bash
    git clone https://github.com/joelsonsartori/HydroMassNet.git
    cd HydroMassNet
    ```

2. Instale las dependencias usando Poetry:
    ```bash
    poetry install
    ```

#### Usando pip

1. Clone el repositorio:
    ```bash
    git clone https://github.com/joelsonsartori/HydroMassNet.git
    cd HydroMassNet
    ```

2. Cree y active un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows use `venv\Scripts\activate`
    ```

3. Instale las dependencias usando pip:
    ```bash
    pip install -r requirements.txt
    ```

### Uso

1. Ejecute el script `bayesian_model.py`:
    ```bash
    poetry run python model/bayesian_model.py  # Si está usando Poetry
    ```

    o

    ```bash
    python model/bayesian_model.py  # Si está usando pip
    ```

2. Verifique los gráficos generados en el directorio del proyecto.

### Estructura del Proyecto

HydroMassNet/
├── data/
│ └── cleaning_data_test.csv
├── func/
│ └── data_preprocessing.py
├── model/
│ ├── bayesian_dense_layer.py
│ ├── bayesian_dense_network.py
│ ├── bayesian_dense_regression.py
│ ├── bayesian_density_network.py
│ └── bayesian_model.py
├── README.md
├── poetry.lock
├── pyproject.toml
└── requirements.txt


### Contacto

Joelson Sartori Junior - [joelsonsartori@gmail.com](mailto:joelsonsartori@gmail.com)
