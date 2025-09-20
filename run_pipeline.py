# file: ./run_pipeline.py
import os
import subprocess
import yaml
import pandas as pd
from src.hydromassnet.preprocess import clean_and_feature_engineer
from src.hydromassnet.plotting import plot_all

def run_command(command):
    """Executa um comando no shell e verifica por erros."""
    print(f"--- Executando: {' '.join(command)} ---")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Erro ao executar o comando: {' '.join(command)}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError("A execução do pipeline falhou.")
    print(result.stdout)

def main():
    """Orquestra o pipeline completo de pré-processamento, treino e avaliação."""
    print("--- Carregando configuração de 'config.yaml' ---")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_data_path = config['paths']['processed_data']
    if not os.path.exists(processed_data_path):
        print(f"Arquivo de dados processados não encontrado em '{processed_data_path}'.")
        clean_and_feature_engineer(config)
    else:
        print("Arquivo de dados processados já existe. Pulando a etapa de pré-processamento.")

    models_to_run = config['models'].keys()
    predictions = {}

    for model_name in models_to_run:
        print(f"\n{'='*20} PROCESSANDO MODELO: {model_name.upper()} {'='*20}")
        run_command(['poetry', 'run', 'python', 'train.py', '--model', model_name])
        run_command(['poetry', 'run', 'python', 'evaluate.py', '--model', model_name])

        prediction_path = os.path.join(config['paths']['results'], f'{model_name}_predictions.csv')
        if os.path.exists(prediction_path):
            predictions[model_name] = pd.read_csv(prediction_path)
        else:
            print(f"AVISO: Arquivo de predição para '{model_name}' não foi encontrado.")

    print("\n--- Gerando gráficos de avaliação ---")
    if predictions:
        plot_all(predictions, config)
        print(f"\nGráficos salvos em: {config['paths']['plots']}")
    else:
        print("Nenhuma predição foi gerada. Gráficos não podem ser criados.")

    print("\n--- Pipeline concluído com sucesso! ---")

if __name__ == "__main__":
    main()
