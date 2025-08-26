import os
import subprocess
import pandas as pd
import itertools
import json

def run_training_job(model_name: str, config: dict, experiment_id: int, feature_set: list):
    """Executa o script de treino como um módulo para resolver problemas de import."""
    command = [
        'python', '-m', 'src.train',
        '--model_name', model_name,
        '--config', json.dumps(config),
        '--experiment_id', str(experiment_id),
        '--features', ",".join(feature_set)
    ]

    print(f"\n--- Iniciando Experimento {experiment_id}: {model_name.upper()} ---")
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print("### ERRO NO TREINAMENTO ###")
        print(result.stderr)
        return None

    try:
        output_json = result.stdout.strip().split('\n')[-1]
        performance = json.loads(output_json)
        return performance
    except (json.JSONDecodeError, IndexError):
        print(f"### ERRO AO PROCESSAR RESULTADO DO TREINAMENTO ###")
        print(f"Saída do script: {result.stdout}")
        return None

def run_hyperparameter_search_for_feature_set(feature_set: list, config: dict):
    search_space = config['hyperparameter_search']
    all_best_params = {}

    for model_name, params_grid in search_space.items():
        keys, values = zip(*params_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"\n{'#'*80}\nIniciando busca para o modelo: {model_name.upper()} com {len(feature_set)} features\n{'#'*80}")

        results = []
        for i, params in enumerate(experiments):
            performance = run_training_job(model_name, params.copy(), i, feature_set)
            if performance:
                params.update(performance)
                results.append(params)

        if not results:
            print(f"Nenhum resultado obtido para {model_name}. Pulando.")
            continue

        results_df = pd.DataFrame(results)
        best_params = results_df.loc[results_df['val_mae'].idxmin()].to_dict()
        all_best_params[model_name] = best_params

        print(f"\n* Busca para {model_name.upper()} concluída. Melhores Hiperparâmetros Encontrados:")
        print(pd.Series(best_params).to_string())
        print("*"*80)

    return all_best_params
