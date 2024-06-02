## Português

[![English](https://img.shields.io/badge/lang-English-blue)](README.md)
[![Español](https://img.shields.io/badge/lang-Español-red)](README.es.md)

# HydroMassNet

HydroMassNet é um projeto voltado para a criação de redes neurais Bayesianas que identificam a porcentagem da massa de hidrogênio neutro em galáxias.

### Pré-requisitos

- Python 3.8 ou superior
- Poetry
- TensorFlow
- TensorFlow Probability
- Scikit-learn
- Matplotlib
- Seaborn

### Instalação

Você pode instalar as dependências usando Poetry ou pip.

#### Usando Poetry

1. Clone o repositório:
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
    source venv/bin/activate  # No Windows use `venv\Scripts\activate`
    ```

3. Instale as dependências usando pip:
    ```bash
    pip install -r requirements.txt
    ```

### Uso

1. Execute o script `bayesian_model.py`:
    ```bash
    poetry run python model/bayesian_model.py  # Se estiver usando Poetry
    ```

    ou

    ```bash
    python model/bayesian_model.py  # Se estiver usando pip
    ```

2. Verifique os gráficos gerados no diretório do projeto.

### Estrutura do Projeto

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


### Contato

Joelson Sartori Junior - [joelsonsartori@gmail.com](mailto:joelsonsartori@gmail.com)
