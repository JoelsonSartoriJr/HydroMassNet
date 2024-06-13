## Español

[![English](https://img.shields.io/badge/lang-English-blue)](README.md)
[![Português](https://img.shields.io/badge/lang-Português-green)](README.pt.md)

# HydroMassNet

HydroMassNet es un proyecto dirigido a la creación de redes neuronales Bayesianas para identificar el porcentaje de masa de hidrógeno neutro en galaxias.

Requisitos
Python 3.8 o superior
Poetry
TensorFlow
TensorFlow Probability
Scikit-learn
Matplotlib
Seaborn
Instalación
Puedes instalar las dependencias usando Poetry o pip.

Usando Poetry

Clona el repositorio:
    ```bash
    git clone https://github.com/joelsonsartori/HydroMassNet.git
    cd HydroMassNet
    ```

2. Instala las dependencias usando Poetry:
    ```bash
    poetry install
    ```

#### Usando pip

1. Clona el repositorio:
    ```bash
    git clone https://github.com/joelsonsartori/HydroMassNet.git
    cd HydroMassNet
    ```

2. Crea y activa un entorno virtual (opcional pero recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Instala las dependencias usando pip:
    ```bash
    pip install -r requirements.txt
    ```

### Uso

1. Ejecuta el script bayesian_model.py:
    ```bash
    poetry run python bayesian_model.py  # If using Poetry
    ```

    o

    ```bash
    python bayesian_model.py  # If using pip
    ```
    
2. Verifica los gráficos generados en el directorio del proyecto.
    ```bash
    poetry run python baseline.py  # If using Poetry
    ```

    o

    ```bash
    python baseline.py  # If using pip
    ```

3.  Ejecuta el script vanilla.py:
    ```bash
    poetry run python vanilla.py  # If using Poetry
    ```

    o

    ```bash
    python vanilla.py  # If using pip
    ```

4. Verifica los gráficos generados en el directorio del proyecto.

### Contato

Joelson Sartori Junior - [joelsonsartori@gmail.com](mailto:joelsonsartori@gmail.com)
