import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import norm

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.usetex": False,            # força não usar LaTeX
    "mathtext.fontset": "dejavusans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "axes.formatter.use_mathtext": True,
    "figure.constrained_layout.use": True
})

def plot_training_histories(histories, config):
    """
    Plota as curvas de Loss e MAE para todos os modelos.

    Parameters
    ----------
    histories : dict
        Dicionário com o histórico de treinamento de cada modelo.
    config : dict
        Configurações do projeto.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Histórico de Treinamento', fontsize=16)

    for model_name, history in histories.items():
        axes[0].plot(history['loss'], label=f'{model_name.upper()} Treino')
        axes[0].plot(history['val_loss'], label=f'{model_name.upper()} Validação', linestyle='--')
    axes[0].set_title('Curvas de Perda (Loss)')
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_yscale('log')

    for model_name, history in histories.items():
        if 'mae' in history.columns:
            axes[1].plot(history['mae'], label=f'{model_name.upper()} Treino')
            axes[1].plot(history['val_mae'], label=f'{model_name.upper()} Validação', linestyle='--')
    axes[1].set_title('Mean Absolute Error (MAE)')
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('MAE')
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(config['paths']['plots'], 'training_histories.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

def plot_predictions_overview(predictions, config):
    """
    Gera um gráfico comparativo de 'valor real vs. predito' para todos os modelos.

    Parameters
    ----------
    predictions : dict
        Dicionário com as predições de cada modelo.
    config : dict
        Configurações do projeto.
    """
    sns.set_theme(style="whitegrid")
    num_models = len(predictions)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5), sharex=True, sharey=True)
    fig.suptitle('Comparação de Modelos: Valor Real vs. Predito (Média)', fontsize=16)

    for i, (model_name, df) in enumerate(predictions.items()):
        ax = axes[i]
        y_true = df['y_true']
        y_pred = df['y_pred_mean']

        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        ax.scatter(y_true, y_pred, alpha=0.3, s=10)

        ax.set_title(f"{model_name.upper()}\nR²={r2:.3f}, RMSE={rmse:.3f}")
        ax.set_xlabel('Valor Real (escalonado)')
        if i == 0:
            ax.set_ylabel('Valor Predito (escalonado)')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(config['paths']['plots'], 'predictions_overview.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

def plot_confidence_intervals(predictions, config):
    """ Plota as predições com seus intervalos de confiança para modelos bayesianos.

        Parameters
        ----------
        predictions : dict
            Dicionário com as predições de cada modelo.
        config : dict
            Configurações do projeto.
    """
    sns.set_theme(style="white")
    bayesian_models = {k: v for k, v in predictions.items() if k in ['bnn', 'dbnn']}
    if not bayesian_models:
        print("Nenhum modelo bayesiano encontrado para plotar intervalos de confiança.")
        return

    num_models = len(bayesian_models)
    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 6), sharex=True, sharey=True)
    if num_models == 1:
        axes = [axes]
    fig.suptitle('Predições com Intervalo de Confiança (95%)', fontsize=16)

    for i, (model_name, df) in enumerate(bayesian_models.items()):
        ax = axes[i]
        df_sorted = df.sort_values(by='y_true').reset_index()

        y_true = df_sorted['y_true']
        y_pred_mean = df_sorted['y_pred_mean']
        y_pred_std = df_sorted['y_pred_std']

        ci = 1.96 * y_pred_std

        ax.scatter(y_true, y_pred_mean, color='blue', s=5, alpha=0.6, label='Predição Média')
        ax.fill_between(y_true, (y_pred_mean - ci), (y_pred_mean + ci), color='blue', alpha=0.2, label='Intervalo de Confiança (95%)')

        ax.set_title(model_name.upper())
        ax.set_xlabel('Valor Real (escalonado)')
        if i == 0:
            ax.set_ylabel('Valor Predito (escalonado)')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(config['paths']['plots'], 'confidence_intervals.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

def plot_all(predictions, histories, config):
    """ Função principal para chamar todas as rotinas de plotagem.

        Parameters
        ----------
        predictions : dict
            Dicionário com as predições de cada modelo.
        histories : dict
            Dicionário com o histórico de treinamento de cada modelo.
        config : dict
            Configurações do projeto.
    """
    print("--- Gerando gráficos de treinamento ---")
    plot_training_histories(histories, config)
    print("--- Gerando gráficos de predição (geral) ---")
    plot_predictions_overview(predictions, config)
    print("--- Gerando gráficos de intervalo de confiança ---")
    plot_confidence_intervals(predictions, config)

def plot_metrics(elbo, mae, r2, model_name, save_dir):
    """ Plota as métricas de ELBO, MAE e R2 para um único modelo.

        Parameters
        ----------
        elbo : np.ndarray
            Array com a perda ELBO por época.
        mae : np.ndarray
            Array com o MAE por época.
        r2 : np.ndarray
            Array com o R2 por época.
        model_name : str
            Nome do modelo.
        save_dir : str
            Diretório para salvar o gráfico.
    """
    plt.figure(figsize=(21, 7))

    plt.subplot(1, 3, 1)
    plt.plot(elbo)
    plt.title(f'{model_name} - ELBO per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')

    plt.subplot(1, 3, 2)
    plt.plot(mae)
    plt.title(f'{model_name} - MAE per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')

    plt.subplot(1, 3, 3)
    plt.plot(r2)
    plt.title(f'{model_name} - R2 Score per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_metrics.png')
    plt.savefig(save_path)
    print(f"Gráfico de métricas salvo em: {save_path}")
    plt.close()

def plot_true_vs_pred_density(df, metrics, save_path):
    """ Gera um 2D density plot (hexbin) com métricas de performance embutidas.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contendo as colunas 'y_true' e 'y_pred_mean'.
        metrics : dict
            Dicionário com as métricas 'mae', 'rmse' e 'r2'.
        save_path : str
            Caminho para salvar o gráfico.
    """
    true_col, pred_col = 'y_true', 'y_pred_mean'

    fig, ax = plt.subplots(figsize=(5, 5))

    min_val = min(df[true_col].min(), df[pred_col].min()) * 0.98
    max_val = max(df[true_col].max(), df[pred_col].max()) * 1.02

    hb = ax.hexbin(df[true_col], df[pred_col], gridsize=50, cmap='viridis', mincnt=1)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.2, label='Linha de Identidade')

    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label('Contagem de Galáxias')

    metrics_text = (f"MAE = {metrics['mae']:.3f}\n"
                    f"RMSE = {metrics['rmse']:.3f}\n"
                    f"$R^2$ = {metrics['r2']:.3f}")
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    ax.set_xlabel(r'Valor Verdadeiro: $\log_{10}(M_{\mathrm{HI}})$ [dex]')
    ax.set_ylabel(r'Valor Predito: $\log_{10}(M_{\mathrm{HI}})$ [dex]')
    ax.set_title('Avaliação do Modelo no Conjunto de Teste')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower right')

    print(f"Salvando gráfico de densidade em: {save_path}")
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

def plot_residuals_distribution(df, metrics, save_path):
    """ Plota a distribuição dos resíduos (erros) do modelo.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contendo as colunas 'y_true' e 'y_pred_mean'.
        metrics : dict
            Dicionário com as métricas.
        save_path : str
            Caminho para salvar o gráfico.
    """
    residuals = df['y_true'] - df['y_pred_mean']

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(residuals, kde=True, ax=ax, stat='density', bins=40)

    mu, std = residuals.mean(), residuals.std()
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k--', linewidth=1.5, label='Ajuste Gaussiano')

    ax.axvline(mu, color='r', linestyle='--', linewidth=1, label=f'Média = {mu:.3f}')

    ax.set_xlabel(r'Resíduo ($\mathrm{Verdadeiro} - \mathrm{Predito}$)')
    ax.set_ylabel('Densidade')
    ax.set_title('Distribuição dos Resíduos do Modelo')
    ax.legend()

    print(f"Salvando gráfico de resíduos em: {save_path}")
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

def plot_posterior_predictive_samples(df, n_samples=5, save_path=None):
    """ Plota a distribuição preditiva para amostras individuais do conjunto de teste.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contendo 'y_true', 'y_pred_mean' e 'y_pred_std'.
        n_samples : int, optional
            Número de amostras para plotar. O padrão é 5.
        save_path : str, optional
            Caminho para salvar o gráfico. O padrão é None.
    """
    samples = df.sample(n_samples, random_state=42)

    fig, axes = plt.subplots(n_samples, 1, figsize=(7, 1.5 * n_samples), sharex=True)
    if n_samples == 1: axes = [axes]

    for i, (_, row) in enumerate(samples.iterrows()):
        y_true, y_mean, y_std = row['y_true'], row['y_pred_mean'], row['y_pred_std']

        x = np.linspace(y_mean - 4*y_std, y_mean + 4*y_std, 500)
        y_pdf = norm.pdf(x, loc=y_mean, scale=y_std)

        axes[i].plot(x, y_pdf, color='C0', label='Distribuição Preditiva')
        axes[i].fill_between(x, y_pdf, color='C0', alpha=0.2)
        axes[i].axvline(y_true, color='C3', linestyle='--', label=f'Verdadeiro: {y_true:.2f}')

        axes[i].set_ylabel('Densidade')
        axes[i].legend()
        axes[i].set_yticklabels([])
        axes[i].tick_params(axis='y', length=0)

    axes[-1].set_xlabel(r'$\log_{10}(M_{\mathrm{HI}})$ [dex]')
    fig.suptitle('Amostras da Distribuição Preditiva Posterior', y=1.02)

    if save_path:
        print(f"Salvando gráfico de distribuições preditivas em: {save_path}")
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
