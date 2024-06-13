## Português

[![English](https://img.shields.io/badge/lang-English-blue)](README.md)
[![Español](https://img.shields.io/badge/lang-Español-red)](README.es.md)

# HydroMassNet

HydroMassNet é um projeto voltado para a criação de redes neurais Bayesianas para identificar a porcentagem de massa de hidrogênio neutro em galáxias.

Pré-requisitos
Python 3.8 ou superior
Poetry
TensorFlow
TensorFlow Probability
Scikit-learn
Matplotlib
Seaborn
Instalação
Você pode instalar as dependências usando Poetry ou pip.

Usando Poetry

Clone o repositório:
    ```bash
    git clone https://github.com/joelsonsartori/HydroMassNet.git
    cd HydroMassNet
    ```

2. Instale as dependências usando o Poetry:
    ```bash
    poetry install
    ```

#### Usando pip

1. Clone o repositório:
    ```bash
    git clone https://github.com/joelsonsartori/HydroMassNet.git
    cd HydroMassNet
    ```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Instale as dependências usando pip:
    ```bash
    pip install -r requirements.txt
    ```

### Uso

1. Execute o script bayesian_model.py:
    ```bash
    poetry run python bayesian_model.py  # If using Poetry
    ```

    ou

    ```bash
    python bayesian_model.py  # If using pip
    ```
    
2. Execute o script baseline.py:
    ```bash
    poetry run python baseline.py  # If using Poetry
    ```

    ou

    ```bash
    python baseline.py  # If using pip
    ```

3.  Execute o script vanilla.py:
    ```bash
    poetry run python vanilla.py  # If using Poetry
    ```

    ou

    ```bash
    python vanilla.py  # If using pip
    ```

4. Verifique os gráficos gerados no diretório do projeto.

### Contato

Joelson Sartori Junior - [joelsonsartori@gmail.com](mailto:joelsonsartori@gmail.com)
