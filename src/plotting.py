# file: ./src/plotting.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import glob
import json

def plot_champion_histories(results_dir: str, plots_dir: str):
    """
    Plota as curvas de aprendizado de todos os modelos campeões.
    """
    history_files = glob.glob(os.path.join(results_dir, '*_history.csv'))
    if not history_files:
        print("Aviso: Nenhum arquivo de histórico encontrado.")
        return

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)

    for file_path in history_files:
        model_name = os.path.basename(file_path).replace('_history.csv', '').replace('champion_', '')
        history_df = pd.read_csv(file_path)

        loss_col, val_loss_col = 'loss', 'val_loss'
        if loss_col in history_df.columns and val_loss_col in history_df.columns:
            axes[0].plot(history_df[loss_col], label=f'{model_name} (treino)')
            axes[0].plot(history_df[val_loss_col], label=f'{model_name} (val)', linestyle='--')

        if 'mae' in history_df.columns and 'val_mae' in history_df.columns:
            axes[1].plot(history_df['mae'], label=f'{model_name} (treino)')
            axes[1].plot(history_df['val_mae'], label=f'{model_name} (val)', linestyle='--')

    axes[0].set_title('Histórico de Perda (Loss)', fontsize=16)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend()
    axes[0].set_yscale('log')

    axes[1].set_title('Histórico de Mean Absolute Error (MAE)', fontsize=16)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].legend()

    fig.suptitle('Comparativo das Curvas de Aprendizado', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(plots_dir, "champion_learning_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráfico de curvas de aprendizado salvo em: {save_path}")

def plot_predictions_analysis(results_dir: str, plots_dir: str):
    """
    Gera gráficos de dispersão (real vs. predito) e de resíduos.
    """
    prediction_files = glob.glob(os.path.join(results_dir, '*_predictions.csv'))
    if not prediction_files:
        return

    num_plots = len(prediction_files)
    fig, axes = plt.subplots(2, num_plots, figsize=(8 * num_plots, 12), sharex='col', squeeze=False)

    for i, file_path in enumerate(prediction_files):
        model_name = os.path.basename(file_path).replace('_predictions.csv', '').replace('champion_', '')
        preds_df = pd.read_csv(file_path)
        preds_df['residuals'] = preds_df['true'] - preds_df['predicted']

        sns.scatterplot(data=preds_df, x='true', y='predicted', ax=axes[0, i], alpha=0.5)
        limits = [preds_df['true'].min(), preds_df['true'].max()]
        axes[0, i].plot(limits, limits, color='red', linestyle='--', linewidth=2, label='Ideal')
        axes[0, i].set_title(f'Predito vs. Real - {model_name.upper()}', fontsize=16)
        axes[0, i].set_xlabel('Valor Real', fontsize=12)
        axes[0, i].set_ylabel('Valor Predito', fontsize=12)
        axes[0, i].grid(True)

        sns.scatterplot(data=preds_df, x='predicted', y='residuals', ax=axes[1, i], alpha=0.5)
        axes[1, i].axhline(0, color='red', linestyle='--')
        axes[1, i].set_title(f'Análise de Resíduos - {model_name.upper()}', fontsize=16)
        axes[1, i].set_xlabel('Valor Predito', fontsize=12)
        axes[1, i].set_ylabel('Resíduos (Real - Predito)', fontsize=12)
        axes[1, i].grid(True)

    fig.suptitle('Análise de Predições no Conjunto de Teste', fontsize=20, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(plots_dir, "predictions_analysis_plots.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráficos de análise de predições salvos em: {save_path}")

def plot_final_metrics_comparison(results_dir: str, plots_dir: str):
    """
    Gera um gráfico de barras comparando as métricas finais (MAE e R²).
    """
    metrics_files = glob.glob(os.path.join(results_dir, '*_metrics.json'))
    if not metrics_files:
        return

    metrics_list = []
    for file_path in metrics_files:
        model_name = os.path.basename(file_path).replace('_metrics.json', '').replace('champion_', '')
        with open(file_path, 'r') as f:
            data = json.load(f)
            metrics_list.append({'model': model_name.upper(), 'MAE': data['mae'], 'R² Score': data['r2_score']})

    metrics_df = pd.DataFrame(metrics_list)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    metrics_df.set_index('model').plot(kind='bar', ax=ax, rot=0)
    ax.set_title('Comparativo de Métricas de Avaliação Finais', fontsize=16)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_xlabel('Modelo', fontsize=12)
    ax.grid(axis='y', linestyle='--')

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    save_path = os.path.join(plots_dir, "final_metrics_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Gráfico de comparação de métricas salvo em: {save_path}")

def main():
    """
    Ponto de entrada para executar todas as plotagens.
    """
    parser = argparse.ArgumentParser(description="Gera gráficos a partir de resultados de treinamento.")
    parser.add_argument('--results_dir', type=str, required=True, help='Diretório com os arquivos de resultados dos campeões.')
    args = parser.parse_args()

    plots_dir = os.path.join(args.results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_champion_histories(args.results_dir, plots_dir)
    plot_predictions_analysis(args.results_dir, plots_dir)
    plot_final_metrics_comparison(args.results_dir, plots_dir)

if __name__ == "__main__":
    main()
