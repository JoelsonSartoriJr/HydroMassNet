import yaml
import json
import os
import pandas as pd
from datetime import datetime

# CORREÇÃO: Alterado para importação relativa
from .feature_selection import run_feature_selection
from .hyperparameter_search import run_hyperparameter_search_for_feature_set

def save_champion_config(best_params: dict, features: list, model_name: str, config_path: str):
    """Salva a configuração do modelo campeão em um arquivo YAML."""
    champion_config = {
        'champion_model': {
            'model': model_name,
            'features': features,
            **best_params
        }
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(champion_config, f, default_flow_style=False, sort_keys=False)
    print(f"\n--- Configuração do campeão salva em: {config_path} ---")

if __name__ == '__main__':
    # Carrega a configuração principal
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 1. Seleção de Features
    print("\n" + "#"*30 + " ETAPA 1: SELEÇÃO DE FEATURES " + "#"*30)
    best_feature_set = run_feature_selection(config)

    # 2. Busca de Hiperparâmetros
    print("\n" + "#"*25 + " ETAPA 2: BUSCA DE HIPERPARÂMETROS " + "#"*25)
    best_params_per_model = run_hyperparameter_search_for_feature_set(best_feature_set, config)

    # 3. Determina o Campeão Final e Salva a Configuração
    print("\n" + "#"*32 + " ETAPA 3: CAMPEÃO FINAL " + "#"*33)
    champion_model_name = None
    best_mae = float('inf')

    for model, params in best_params_per_model.items():
        if params['val_mae'] < best_mae:
            best_mae = params['val_mae']
            champion_model_name = model

    if champion_model_name:
        champion_params = best_params_per_model[champion_model_name]
        print(f"O modelo campeão é: {champion_model_name.upper()} com Val_MAE: {best_mae:.4f}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        champion_config_path = os.path.join(config['paths']['reports'], f'champion_config_{timestamp}.yaml')
        save_champion_config(champion_params, best_feature_set, champion_model_name, champion_config_path)
    else:
        print("Não foi possível determinar um modelo campeão.")

    print("\n" + "#"*30 + " OTIMIZAÇÃO COMPLETA FINALIZADA " + "#"*29)
