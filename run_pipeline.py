import os
import subprocess
import pandas as pd
import yaml
from src.plotting import plot_all
from src.preprocess import clean_and_feature_engineer

def run_command(command):
    """
    Executa um comando no shell e verifica por erros.

    Parameters
    ----------
    command : list
        Lista de strings do comando a ser executado.
    """
    print(f"--- Executando: {' '.join(command)} ---")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"Erro ao executar o comando: {' '.join(command)}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError("A execução do pipeline falhou.")
    print(result.stdout)

def main():
    """
    Orquestra o pipeline completo de pré-processamento, treinamento e avaliação.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    processed_data_path = config['paths']['processed_data']
    if not os.path.exists(processed_data_path):
        print(f"Arquivo de dados processados não encontrado em '{processed_data_path}'.")
        clean_and_feature_engineer(config)
    else:
        print("Arquivo de dados processados já existe. Pulando a etapa de pré-processamento.")

    models_to_run = ['bnn', 'dbnn', 'vanilla']
    predictions = {}
    histories = {}

    for model_name in models_to_run:
        run_command(['poetry', 'run', 'python', 'src/train.py', '--model', model_name])
        run_command(['poetry', 'run', 'python', 'src/evaluate.py', '--model', model_name])

        prediction_path = os.path.join(config['paths']['saved_models'], f'{model_name}_predictions.csv')
        history_path = os.path.join(config['paths']['saved_models'], f'{model_name}_history.csv')

        predictions[model_name] = pd.read_csv(prediction_path)
        histories[model_name] = pd.read_csv(history_path)

    run_command(['poetry', 'run', 'python', 'src/baseline.py'])
    baseline_prediction_path = os.path.join(config['paths']['saved_models'], 'baseline_predictions.csv')
    predictions['baseline'] = pd.read_csv(baseline_prediction_path)

    plot_all(predictions, histories, config)

    print(f"\nGráficos salvos em: {config['paths']['plots']}")
    print("--- Pipeline concluído com sucesso! ---")

if __name__ == "__main__":
    main()
