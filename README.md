## English

[![Português](https://img.shields.io/badge/lang-Português-green)](README.pt.md)
[![Español](https://img.shields.io/badge/lang-Español-red)](README.es.md)

# HydroMassNet

HydroMassNet is a project aimed at creating Bayesian neural networks to identify the percentage of neutral hydrogen mass in galaxies.

### Prerequisites

- Python 3.8 or higher
- Poetry
- TensorFlow
- TensorFlow Probability
- Scikit-learn
- Matplotlib
- Seaborn

### Installation

You can install the dependencies using Poetry or pip.

#### Using Poetry

1. Clone the repository:
    ```bash
    git clone https://github.com/joelsonsartori/HydroMassNet.git
    cd HydroMassNet
    ```

2. Install dependencies using Poetry:
    ```bash
    poetry install
    ```

#### Using pip

1. Clone the repository:
    ```bash
    git clone https://github.com/joelsonsartori/HydroMassNet.git
    cd HydroMassNet
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the `bayesian_model.py` script:
    ```bash
    poetry run python bayesian_model.py  # If using Poetry
    ```

    or

    ```bash
    python bayesian_model.py  # If using pip
    ```

2. Check the generated plots in the project directory.

### Project Structure

.
├── BNN_Accuracy.png
├── BNN_Elbo.png
├── BNN_Mae.png
├── BNN_model.h5
├── BNN_model.keras
├── DBNN_Accuracy.png
├── DBNN_Elbo.png
├── DBNN_Mae.png
├── DBNN_model.h5
├── DBNN_model.keras
├── LICENSE
├── README.es.md
├── README.md
├── README.pt.md
├── Residuo.png
├── baseline.py
├── bayesian_model.py
├── class
│ └── lr_history.py
├── data
│ ├── cleaning_data.csv
│ ├── cleaning_data_test.csv
│ └── data.csv
├── func
│ ├── pycache
│ │ ├── data_preprocessing.cpython-312.pyc
│ │ ├── plot.cpython-312.pyc
│ │ ├── train.cpython-312.pyc
│ │ └── utils.cpython-312.pyc
│ ├── data_preprocessing.py
│ ├── plot.py
│ ├── set_seed.py
│ ├── train.py
│ └── utils.py
├── model
│ ├── pycache
│ │ ├── bayesian_dense_layer.cpython-312.pyc
│ │ ├── bayesian_dense_network.cpython-312.pyc
│ │ ├── bayesian_dense_regression.cpython-312.pyc
│ │ └── bayesian_density_network.cpython-312.pyc
│ ├── bayesian_dense_layer.py
│ ├── bayesian_dense_network.py
│ ├── bayesian_dense_regression.py
│ └── bayesian_density_network.py
├── model_accuracy.png
├── poetry.lock
├── pred_bnn.png
├── predict.py
├── pyproject.toml
├── realVSpreditoBNN.png
├── realVSpreditoDBNN.png
├── requirements.txt
└── vanilla.py


### Contact

Joelson Sartori Junior - [joelsonsartori@gmail.com](mailto:joelsonsartori@gmail.com)
