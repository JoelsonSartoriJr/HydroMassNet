import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_results(csv_path):
    df = pd.read_csv(csv_path)

    # Agrupa por número de features e pega o melhor resultado (maior R²)
    best_per_n = df.loc[df.groupby('num_features')['r2'].idxmax()]

    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot R²
    color = 'tab:blue'
    ax1.set_xlabel('Número de Features', fontsize=14)
    ax1.set_ylabel('R² (R-squared)', color=color, fontsize=14)
    ax1.plot(best_per_n['num_features'], best_per_n['r2'], 'o-', color=color, label='Melhor R²')
    ax1.tick_params(axis='y', labelcolor=color)

    # Cria um segundo eixo Y para o RMSE
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('RMSE (Root Mean Squared Error)', color=color, fontsize=14)
    ax2.plot(best_per_n['num_features'], best_per_n['rmse'], 's--', color=color, label='Melhor RMSE')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance do Modelo vs. Número de Features', fontsize=16)
    fig.tight_layout()

    # Salva o gráfico
    output_path = 'results/plots/feature_selection_performance.png'
    plt.savefig(output_path)
    print(f"Gráfico salvo em: {output_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota os resultados da seleção de features.")
    parser.add_argument("csv_path", type=str, help="Caminho para o arquivo CSV de resultados.")
    args = parser.parse_args()
    plot_results(args.csv_path)
