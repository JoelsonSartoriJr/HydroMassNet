# file: ./main.py
import os
import yaml
import json
from datetime import datetime
from src.utils.commands import run_command

def main():
    """
    Ponto de entrada principal para orquestrar o pipeline completo.
    """
    print(">>> INICIANDO O PIPELINE COMPLETO DE OTIMIZAÇÃO <<<")

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # --- Passo 1: Seleção de Features ---
    print("\n" + "="*80)
    print("PASSO 1: SELEÇÃO DOS MELHORES CONJUNTOS DE FEATURES")
    print("="*80)
    run_command(['poetry', 'run', 'python', '-m', 'src.feature_selection'])

    # --- Passo 2: Busca de Hiperparâmetros ---
    print("\n" + "="*80)
    print("PASSO 2: BUSCA DE HIPERPARÂMETROS PARA CADA CONJUNTO DE FEATURES")
    print("="*80)
    run_command(['poetry', 'run', 'python', '-m', 'src.hyperparameter_search'])

    champion_config_path = os.path.join(config['paths']['search_results'], 'champion_config.yaml')
    if not os.path.exists(champion_config_path):
        raise FileNotFoundError("champion_config.yaml não foi encontrado. A busca de hiperparâmetros falhou.")

    with open(champion_config_path, 'r', encoding='utf-8') as f:
        champion_configs = yaml.safe_load(f)

    # --- Passo 3: Treinamento e Avaliação dos Campeões ---
    print("\n" + "="*80)
    print("PASSO 3: TREINAMENTO E AVALIAÇÃO DOS MODELOS CAMPEÕES")
    print("="*80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    champions_dir = os.path.join(config['paths']['champion_results'], timestamp)
    os.makedirs(champions_dir, exist_ok=True)
    print(f"Diretório de resultados dos campeões: {champions_dir}")

    for name, model_cfg in champion_configs.items():
        model_type = name.replace('champion_', '')
        print(f"\n{'#'*80}\n### Processando Campeão: {name.upper()} ###\n{'#'*80}")
        save_path = os.path.join(champions_dir, f"champion_{model_type}")
        run_command([
            'poetry', 'run', 'python', '-m', 'src.train',
            '--model_type', model_type,
            '--model_config', json.dumps(model_cfg),
            '--save_path', save_path
        ])
        run_command([
            'poetry', 'run', 'python', '-m', 'src.evaluate',
            '--model_type', model_type,
            '--model_config', json.dumps(model_cfg),
            '--weights_path', f"{save_path}.weights.h5",
            '--output_dir', champions_dir
        ])

    # --- Passo 4: Geração de Gráficos ---
    print("\n" + "="*80)
    print("PASSO 4: GERAÇÃO DOS GRÁFICOS FINAIS")
    print("="*80)
    run_command([
        'poetry', 'run', 'python', '-m', 'src.plotting',
        '--results_dir', champions_dir
    ])

    print("\n>>> PIPELINE COMPLETO CONCLUÍDO COM SUCESSO! <<<")
    print(f"Resultados finais disponíveis em: {champions_dir}")

if __name__ == "__main__":
    main()
