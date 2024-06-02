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
    poetry run python model/bayesian_model.py  # If using Poetry
    ```

    or

    ```bash
    python model/bayesian_model.py  # If using pip
    ```

2. Check the generated plots in the project directory.

### Project Structure

    .
    ├── ...
    ├── docs                    # Documentation files (alternatively `doc`)
    │   ├── TOC.md              # Table of contents
    │   ├── faq.md              # Frequently asked questions
    │   ├── misc.md             # Miscellaneous information
    │   ├── usage.md            # Getting started guide
    │   └── ...                 # etc.
    └── ...


### Contact

Joelson Sartori Junior - [joelsonsartori@gmail.com](mailto:joelsonsartori@gmail.com)
