#!/usr/bin/env python3
"""Helper script to regenerate every plot (incl. color–massa)."""

import os
import yaml
import pandas as pd
from src.hydromassnet.plotting import plot_all, plot_color_stellar_mass_diagram


def _load_predictions(config):
    predictions = {}
    models_cfg = config.get('models', {})
    results_dir = config['paths']['results']

    for model_name in models_cfg:
        path = os.path.join(results_dir, f'{model_name}_predictions.csv')
        if not os.path.exists(path):
            print(f"[aviso] Arquivo de predições não encontrado: {path}")
            continue
        try:
            predictions[model_name] = pd.read_csv(path)
        except Exception as exc:
            print(f"[erro] Falha ao carregar {path}: {exc}")
    return predictions

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('--- carregando predições existentes ---')
predictions = _load_predictions(config)
if predictions:
    print('--- gerando pacotes completos de gráficos ---')
    plot_all(predictions, config)
else:
    print('Nenhuma predição encontrada; plots comparativos serão ignorados.')

print('--- gerando diagrama cor–massa ---')
plot_color_stellar_mass_diagram(config)
print('Todos os gráficos foram gerados em results/plots/.')
