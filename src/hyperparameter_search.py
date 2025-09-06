# file: ./src/hyperparameter_search.py
import os
import yaml
import json
import itertools
import pandas as pd
from datetime import datetime
from src.utils.commands import run_command

def run_training_job(model_type: str, params: dict, experiment_id: int, save_dir: str):
    """
    Executa um job de treinamento e retorna a métrica de performance.

    Args:
        model_type (str): O tipo de modelo ('bnn', 'dbnn', 'vanilla').
        params (dict): Dicionário com os hiperparâmetros para o experimento.
        experiment_id (int): Um identificador único para o experimento.
        save_dir (str): Diretório para salvar os artefatos temporários do job.

    Returns:
        float: O MAE de validação, ou infinito em caso de erro.
    """
    print(f"\n--- Iniciando Experimento {experiment_id}: {model_type.upper()} ---")
    save_path = os.path.join(save_dir, f"{model_type}_exp_{experiment_id}")

    command = [
        'poetry', 'run', 'python', '-m', 'src.train',
        '--model_type', model_type,
        '--model_config', json.dumps(params),
        '--save_path', save_path
    ]
    try:
        output = run_command(command)
        performance_str = output.strip().split('\n')[-1]
        performance = json.loads(performance_str)
        return performance['val_mae']
    except (RuntimeError, json.JSONDecodeError, IndexError) as e:
        print(f"### ERRO no experimento {experiment_id} para {model_type}: {e} ###")
        return float('inf')

def main():
    """
    Orquestra a busca de hiperparâmetros para todos os modelos.
    """
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    search_space = config['hyperparameter_search']
    search_dir = config['paths']['search_results']
    os.makedirs(search_dir, exist_ok=True)

    champion_configs = {}
    full_report = []

    for model_type, grid in search_space.items():
        print(f"\n{'#'*80}\n### Iniciando busca para: {model_type.upper()} ###\n{'#'*80}")

        keys, values = zip(*grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        results = []
        for i, params in enumerate(experiments):
            val_mae = run_training_job(model_type, params, i, search_dir)
            result_row = params.copy()
            result_row['features'] = json.dumps(result_row['features']) # Converte lista para string para o CSV
            result_row['val_mae'] = val_mae
            results.append(result_row)
            full_report.append({'model_type': model_type, **result_row})

        if not results:
            print(f"Nenhum resultado válido para {model_type}. Pulando.")
            continue

        results_df = pd.DataFrame(results)
        best_params_row = results_df.loc[results_df['val_mae'].idxmin()]
        best_params = best_params_row.to_dict()

        best_params['features'] = json.loads(best_params['features'])
        del best_params['val_mae']
        champion_configs[f'champion_{model_type}'] = best_params

        print(f"\n* Melhor configuração para {model_type.upper()}:")
        print(json.dumps(best_params, indent=2))

    report_path = os.path.join(search_dir, f"full_search_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(full_report).to_csv(report_path, index=False)
    print(f"\nRelatório completo da busca salvo em: {report_path}")

    champion_config_path = os.path.join(search_dir, 'champion_config.yaml')
    with open(champion_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(champion_configs, f, default_flow_style=False)
    print(f"Configuração dos campeões salva em: {champion_config_path}")

if __name__ == "__main__":
    main()
