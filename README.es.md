## Español

[![English](https://img.shields.io/badge/lang-English-blue)](README.md)
[![Português](https://img.shields.io/badge/lang-Português-green)](README.pt.md)

HydroMassNet

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

bash
Copiar código
git clone https://github.com/joelsonsartori/HydroMassNet.git
cd HydroMassNet
Instala las dependencias usando Poetry:

bash
Copiar código
poetry install
Usando pip

Clona el repositorio:

bash
Copiar código
git clone https://github.com/joelsonsartori/HydroMassNet.git
cd HydroMassNet
Crea y activa un entorno virtual (opcional pero recomendado):

bash
Copiar código
python -m venv venv
source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
Instala las dependencias usando pip:

bash
Copiar código
pip install -r requirements.txt
Uso
Ejecuta el script bayesian_model.py:

bash
Copiar código
poetry run python bayesian_model.py  # Si usas Poetry
o

bash
Copiar código
python bayesian_model.py  # Si usas pip
Ejecuta el script baseline.py:

bash
Copiar código
poetry run python baseline.py  # Si usas Poetry
o

bash
Copiar código
python baseline.py  # Si usas pip
Ejecuta el script vanilla.py:

bash
Copiar código
poetry run python vanilla.py  # Si usas Poetry
o

bash
Copiar código
python vanilla.py  # Si usas pip
Verifica los gráficos generados en el directorio del proyecto.

Contacto
Joelson Sartori Junior - joelsonsartori@gmail.com
