# file: ./src/feature_selection.py
import os
import yaml
import json
import itertools
import pandas as pd
from datetime import datetime
from src.utils.commands import run_command

def run_feature_eval_job(features: list, config: dict, experiment_id: int):
    """
    Executa um job de treinamento rápido para avaliar um conjunto de features.

    Args:
        features (list): O conjunto de features a ser testado.
        config (dict): O dicionário de configuração principal.
        experiment_id (int): Um identificador único para o experimento.

    Returns:
        float: O MAE de validação para o conjunto de features, ou infinito em caso de erro.
    """
    model_cfg = config['feature_selection']['evaluation_model_config']
    model_cfg['features'] = features

    save_dir = config['paths']['feature_selection_results']
    save_path = os.path.join(save_dir, f"feature_eval_exp_{experiment_id}")

    command = [
        'poetry', 'run', 'python', '-m', 'src.train',
        '--model_type', 'vanilla',
        '--model_config', json.dumps(model_cfg),
        '--save_path', save_path
    ]
    try:
        output = run_command(command)
        performance_str = output.strip().split('\n')[-1]
        performance = json.loads(performance_str)
        return performance['val_mae']
    except (RuntimeError, json.JSONDecodeError, IndexError) as e:
        print(f"### ERRO no experimento de feature selection {experiment_id}: {e} ###")
        return float('inf')

def main():
    """
    Orquestra a seleção do melhor conjunto de features.
    """
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    fs_config = config['feature_selection']
    base_features = fs_config['candidate_features']
    min_features = fs_config['min_features']

    results_dir = config['paths']['feature_selection_results']
    os.makedirs(results_dir, exist_ok=True)

    print(f"Iniciando a busca pelo melhor conjunto de features a partir de {len(base_features)} candidatas.")

    all_combinations = []
    for i in range(min_features, len(base_features) + 1):
        all_combinations.extend(itertools.combinations(base_features, i))

    print(f"Total de {len(all_combinations)} combinações de features a serem testadas.")

    results = []
    for i, features in enumerate(all_combinations):
        feature_list = list(features)
        print(f"\n--- Testando combinação {i+1}/{len(all_combinations)}: {feature_list} ---")
        val_mae = run_feature_eval_job(feature_list, config, i)
        results.append({'features': ','.join(feature_list), 'num_features': len(feature_list), 'val_mae': val_mae})

    results_df = pd.DataFrame(results)
    best_feature_set_row = results_df.loc[results_df['val_mae'].idxmin()]
    best_features = best_feature_set_row['features'].split(',')

    report_path = os.path.join(results_dir, f"feature_selection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(report_path, index=False)

    print("\n" + "*"*80)
    print("Busca de Features Concluída!")
    print(f"Relatório completo salvo em: {report_path}")
    print(f"Melhor conjunto de features encontrado ({len(best_features)} features) com val_mae={best_feature_set_row['val_mae']:.4f}:")
    print(best_features)
    print("*"*80 + "\n")

    for model_type in config['hyperparameter_search']:
        config['hyperparameter_search'][model_type]['features'] = [best_features]

    with open('config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

    print("Arquivo 'config.yaml' foi atualizado com o melhor conjunto de features.")

if __name__ == "__main__":
    main()
