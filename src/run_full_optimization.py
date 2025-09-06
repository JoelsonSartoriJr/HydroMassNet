import os
import yaml
import pandas as pd
from datetime import datetime
import sys
import numpy as np

# Adiciona o diretório raiz ao path para encontrar o pacote 'src'
sys.path.append(os.getcwd())

# 🔧 Corrigido para usar imports de pacote
from src.feature_selection import run_feature_selection
from src.hyperparameter_search import run_hyperparameter_search_for_feature_set
from src.utils.plotting import plot_full_optimization_results


def main():
    """
    Orquestra o pipeline de otimização completo, salvando o melhor resultado para cada tipo de modelo.
    """
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- ESTÁGIO 1: SELEÇÃO DE FEATURES ---
    print("#" * 80 + "\n### ESTÁGIO 1: SELEÇÃO DE FEATURES ###\n" + "#" * 80)
    feature_results_path = run_feature_selection(config)
    if not feature_results_path:
        print("Seleção de features falhou. Abortando.")
        return

    feature_results_df = pd.read_csv(feature_results_path)
    top_n = config['feature_selection']['top_n_to_tune']
    best_feature_sets = [s.split(', ') for s in feature_results_df.head(top_n)['features']]

    print("\n" + "#" * 80 + f"\n### TOP {top_n} CONJUNTOS DE FEATURES ENCONTRADOS ###\n" + "#" * 80)
    [print(f"  {i+1}. {fset}") for i, fset in enumerate(best_feature_sets)]

    # --- ESTÁGIO 2: BUSCA DE HIPERPARÂMETROS ---
    print("\n" + "#" * 80 + f"\n### ESTÁGIO 2: BUSCA DE HIPERPARÂMETROS ###\n" + "#" * 80)
    final_results = []
    for i, fset in enumerate(best_feature_sets):
        print(f"\n--- INICIANDO BUSCA PARA CONJUNTO #{i+1}/{len(best_feature_sets)} | Features: {fset} ---")
        best_params = run_hyperparameter_search_for_feature_set(fset, config)
        for model, params in best_params.items():
            entry = {'model': model, 'feature_set_rank': i + 1, 'features': ", ".join(fset)}
            entry.update(params)
            final_results.append(entry)

    if not final_results:
        print("Busca de hiperparâmetros não produziu resultados. Abortando.")
        return

    # --- ESTÁGIO 3: SALVAR RESULTADOS E CONFIGURAÇÕES DOS CAMPEÕES ---
    final_df = pd.DataFrame(final_results).sort_values(by='val_mae')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(config['paths']['search_results'], f'full_report_{timestamp}.csv')
    final_df.to_csv(report_path, index=False)

    print("\n" + "#" * 80 + "\n### OTIMIZAÇÃO COMPLETA CONCLUÍDA! ###\n" + "#" * 80)
    print(f"Relatório final salvo em: {report_path}")
    print("\n--- MELHOR COMBINAÇÃO GERAL ENCONTRADA ---")
    print(final_df.head(1).to_string())

    # --- LÓGICA ATUALIZADA PARA SALVAR O MELHOR DE CADA MODELO ---
    print("\n--- Salvando melhores configurações para cada tipo de modelo ---")

    champion_configs = {}
    # Encontra o melhor resultado (menor val_mae) para cada grupo de modelo
    best_per_model = final_df.loc[final_df.groupby('model')['val_mae'].idxmin()]

    for _, row in best_per_model.iterrows():
        model_name = row['model']
        print(f"  - Melhor para {model_name.upper()}: val_mae = {row['val_mae']:.4f}")

        # Monta o dicionário de configuração para este campeão
        config_entry = {}
        # Lista de colunas que não são hiperparâmetros
        params_to_exclude = [
            'model', 'feature_set_rank', 'val_loss', 'val_mae',
            'loss', 'mae', 'mse', 'val_mse', 'val_rmse', 'rmse'
        ]

        for key, value in row.items():
            # Só adiciona se for um hiperparâmetro e não for um valor nulo (NaN)
            if key not in params_to_exclude and pd.notna(value):
                # Converte tipos numpy para tipos nativos do Python
                config_entry[key] = value.item() if isinstance(value, np.generic) else value

        config_entry['features'] = [f.strip() for f in row['features'].split(',')]
        champion_configs[f'champion_{model_name}'] = config_entry

    # Salva o arquivo YAML com todas as configurações campeãs
    champion_path = os.path.join(config['paths']['search_results'], 'champion_config.yaml')
    with open(champion_path, 'w') as f:
        yaml.dump(champion_configs, f, sort_keys=False, indent=2)
    print(f"\nConfigurações dos campeões salvas em: {champion_path}")

    # Plotar gráfico final
    plot_path = os.path.join(config['paths']['plots'], f'optimization_summary_{timestamp}.png')
    plot_full_optimization_results(final_df, plot_path)


if __name__ == "__main__":
    main()
