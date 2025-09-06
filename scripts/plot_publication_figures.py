import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# --- Estilo Unificado para Publicação Acadêmica ---
# Inspirado nas diretrizes do arquivo Figure_Plan_and_Draft_Verdict.pdf
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"], # Fonte similar a Times/LaTeX
    "text.usetex": True, # Essencial para formatação LaTeX-style
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

def plot_true_vs_pred_density(df, metrics, save_path):
    """
    Gera um 2D density plot (hexbin) com métricas de performance embutidas.
    """
    true_col, pred_col = 'y_true', 'y_pred_mean'

    fig, ax = plt.subplots(figsize=(5, 5))

    min_val = min(df[true_col].min(), df[pred_col].min()) * 0.98
    max_val = max(df[true_col].max(), df[pred_col].max()) * 1.02

    hb = ax.hexbin(df[true_col], df[pred_col], gridsize=50, cmap='viridis', mincnt=1)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.2, label='Linha de Identidade')

    # Adiciona a barra de cores
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label('Contagem de Galáxias')

    # Adiciona as métricas no gráfico
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

    # Salva em formato vetorial PDF
    print(f"Salvando gráfico de densidade em: {save_path}")
    fig.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

def plot_residuals_distribution(df, metrics, save_path):
    """
    Plota a distribuição dos resíduos (erros) do modelo.
    """
    residuals = df['y_true'] - df['y_pred_mean']

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(residuals, kde=True, ax=ax, stat='density', bins=40)

    # Plota uma gaussiana com a média e o desvio padrão dos resíduos
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
    """
    Plota a distribuição preditiva para amostras individuais do conjunto de teste.
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
