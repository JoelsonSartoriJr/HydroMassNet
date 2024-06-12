## Português

[![English](https://img.shields.io/badge/lang-English-blue)](README.md)
[![Español](https://img.shields.io/badge/lang-Español-red)](README.es.md)

Português



HydroMassNet

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

bash
Copiar código
git clone https://github.com/joelsonsartori/HydroMassNet.git
cd HydroMassNet
Instale as dependências usando o Poetry:

bash
Copiar código
poetry install
Usando pip

Clone o repositório:

bash
Copiar código
git clone https://github.com/joelsonsartori/HydroMassNet.git
cd HydroMassNet
Crie e ative um ambiente virtual (opcional, mas recomendado):

bash
Copiar código
python -m venv venv
source venv/bin/activate  # No Windows use `venv\Scripts\activate`
Instale as dependências usando pip:

bash
Copiar código
pip install -r requirements.txt
Uso
Execute o script bayesian_model.py:

bash
Copiar código
poetry run python bayesian_model.py  # Se estiver usando o Poetry
ou

bash
Copiar código
python bayesian_model.py  # Se estiver usando o pip
Execute o script baseline.py:

bash
Copiar código
poetry run python baseline.py  # Se estiver usando o Poetry
ou

bash
Copiar código
python baseline.py  # Se estiver usando o pip
Execute o script vanilla.py:

bash
Copiar código
poetry run python vanilla.py  # Se estiver usando o Poetry
ou

bash
Copiar código
python vanilla.py  # Se estiver usando o pip
Verifique os gráficos gerados no diretório do projeto.

Contato
Joelson Sartori Junior - joelsonsartori@gmail.com

css
Copiar código

Esse README oferece uma visão geral completa do projeto, incluindo a instalação das dependências usando Poetry ou pip, instruções de uso e informações de contato em português.
