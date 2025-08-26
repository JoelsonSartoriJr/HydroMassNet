import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_selection_results(results_df: pd.DataFrame, output_path: str):
    """
    Plota e salva os resultados da seleção de features (R² e RMSE vs. N de Features).
    """
    if results_df.empty:
        print("DataFrame de resultados da seleção de features está vazio. Pulando plotagem.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Agrupa por número de features e pega o melhor resultado (maior R²) para o plot
    best_per_n = results_df.loc[results_df.groupby('num_features')['r2'].idxmax()]

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
    ax2.set_ylabel('RMSE (Erro)', color=color, fontsize=14)
    ax2.plot(best_per_n['num_features'], best_per_n['rmse'], 's--', color=color, label='Menor RMSE')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance vs. Número de Features', fontsize=16)
    fig.tight_layout()

    plt.savefig(output_path)
    print(f"Gráfico da seleção de features salvo em: {output_path}")
    plt.close()

def plot_full_optimization_results(final_df: pd.DataFrame, output_path: str):
    """
    Plota o resultado final da otimização, mostrando o melhor MAE para cada conjunto de features.
    """
    if final_df.empty:
        print("DataFrame de resultados finais está vazio. Pulando plotagem.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Encontra a melhor performance (menor MAE) para cada rank de feature set
    best_per_set = final_df.loc[final_df.groupby('feature_set_rank')['val_mae'].idxmin()]

    plt.figure(figsize=(12, 7))
    sns.barplot(x='feature_set_rank', y='val_mae', data=best_per_set, palette='viridis', hue='feature_set_rank', dodge=False, legend=False)

    plt.xlabel('Rank do Conjunto de Features', fontsize=14)
    plt.ylabel('Melhor MAE de Validação', fontsize=14)
    plt.title('Performance Final por Conjunto de Features', fontsize=16)
    plt.xticks(ticks=range(len(best_per_set)), labels=[f"Rank {i+1}" for i in range(len(best_per_set))], rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(output_path)
    print(f"Gráfico da otimização completa salvo em: {output_path}")
    plt.close()
