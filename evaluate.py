import argparse
import yaml
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Importa as novas funções de plotagem
from src.publication_plots import (
    plot_true_vs_pred_density,
    plot_residuals_distribution,
    plot_posterior_predictive_samples
)

def evaluate(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    predictions_path = os.path.join(config['paths']['results'], f'{args.model}_predictions.csv')
    if not os.path.exists(predictions_path):
        print(f"ERRO: Arquivo de predições não encontrado em {predictions_path}")
        print(f"Por favor, execute 'python predict.py --model {args.model}' primeiro.")
        return

    df = pd.read_csv(predictions_path)

    # --- 1. Calcular Métricas Completas ---
    y_true = df['y_true']
    y_pred = df['y_pred_mean']

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False),
        'r2': r2_score(y_true, y_pred)
    }

    print("\n--- Métricas de Avaliação para Publicação ---")
    print(f"Modelo: {args.model.upper()}")
    print(f"  MAE  = {metrics['mae']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.4f}")
    print(f"  R²   = {metrics['r2']:.4f}")
    print("-------------------------------------------\n")

    # --- 2. Gerar Gráficos com Qualidade de Publicação ---
    plots_dir = os.path.join(config['paths']['plots'], args.model)
    os.makedirs(plots_dir, exist_ok=True)

    # Gráfico 1: Densidade Verdadeiro vs. Predito (com métricas)
    save_path_density = os.path.join(plots_dir, 'fig_true_vs_pred.pdf')
    plot_true_vs_pred_density(df, metrics, save_path_density)

    # Gráfico 2: Distribuição dos Resíduos
    save_path_residuals = os.path.join(plots_dir, 'fig_residuals_distribution.pdf')
    plot_residuals_distribution(df, metrics, save_path_residuals)

    # Gráfico 3: Amostras da Distribuição Preditiva Posterior
    save_path_posterior = os.path.join(plots_dir, 'fig_posterior_samples.pdf')
    plot_posterior_predictive_samples(df, n_samples=5, save_path=save_path_posterior)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliador de Modelos para HydroMassNet")
    parser.add_argument('--model', type=str, required=True, choices=['bnn', 'dbnn'], help='Qual modelo avaliar.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Caminho para o arquivo de configuração.')
    args = parser.parse_args()
    evaluate(args)
