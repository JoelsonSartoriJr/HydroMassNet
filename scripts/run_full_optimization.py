import os
import yaml
import pandas as pd
from datetime import datetime
import sys

# Adiciona o diretório raiz ao path para encontrar o pacote 'src'
sys.path.append(os.getcwd())
from src.utils.plotting import plot_full_optimization_results
from feature_selection import run_feature_selection
from hyperparameter_search import run_hyperparameter_search_for_feature_set

def main():
    """Orquestra o pipeline de otimização completo."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # --- ESTÁGIO 1: SELEÇÃO DE FEATURES ---
    print("#"*80)
    print("### INICIANDO ESTÁGIO 1: SELEÇÃO DE FEATURES ###")
    print("#"*80)
    feature_results_path = run_feature_selection(config)
    if not feature_results_path or not os.path.exists(feature_results_path):
        print("A seleção de features falhou. Abortando.")
        return

    feature_results_df = pd.read_csv(feature_results_path)
    top_n_to_tune = config['feature_selection']['top_n_to_tune']
    best_feature_sets_str = feature_results_df.head(top_n_to_tune)['features'].tolist()
    best_feature_sets = [s.split(', ') for s in best_feature_sets_str]

    print("\n" + "#"*80)
    print(f"### TOP {top_n_to_tune} CONJUNTOS DE FEATURES ENCONTRADOS ###")
    [print(f"  {i+1}. {fset}") for i, fset in enumerate(best_feature_sets)]
    print("#"*80)

    # --- ESTÁGIO 2: BUSCA DE HIPERPARÂMETROS ---
    print("\n" + "#"*80)
    print("### INICIANDO ESTÁGIO 2: BUSCA DE HIPERPARÂMETROS ###")
    print("#"*80)
    final_results = []
    for i, feature_set in enumerate(best_feature_sets):
        print("\n" + "="*80)
        print(f"--- BUSCA PARA O CONJUNTO DE FEATURES #{i+1}/{top_n_to_tune} ---")
        print(f"--- Features: {feature_set} ---")
        print("="*80)

        best_params_for_set = run_hyperparameter_search_for_feature_set(feature_set, config)

        for model_name, params in best_params_for_set.items():
            result_entry = {'model': model_name, 'feature_set_rank': i + 1, 'features': ", ".join(feature_set)}
            result_entry.update(params)
            final_results.append(result_entry)

    if not final_results:
        print("A busca de hiperparâmetros não produziu resultados. Abortando.")
        return

    final_results_df = pd.DataFrame(final_results).sort_values(by='val_mae', ascending=True)

    results_dir = config['paths']['search_results']
    plots_dir = config['paths']['plots']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_report_path = os.path.join(results_dir, f'full_optimization_report_{timestamp}.csv')
    final_results_df.to_csv(final_report_path, index=False)

    print("\n" + "#"*80)
    print("### OTIMIZAÇÃO COMPLETA CONCLUÍDA! ###")
    print(f"Relatório final salvo em: {final_report_path}")
    print("\n--- MELHOR COMBINAÇÃO GERAL ENCONTRADA ---")
    print(final_results_df.head(1).to_string())
    print("#"*80)

    # --- ESTÁGIO 3: SALVAR CONFIGURAÇÃO CAMPEÃ E PLOTAR RESULTADO FINAL ---
    best_result = final_results_df.iloc[0]
    champion_config = {'champion_model': {}}
    params_to_exclude = ['val_mae', 'val_loss', 'val_rmse', 'val_r2', 'feature_set_rank', 'features']
    for key, value in best_result.items():
        if key not in params_to_exclude:
            champion_config['champion_model'][key] = value.item() if hasattr(value, 'item') else value
    champion_config['champion_model']['features'] = [f.strip() for f in best_result['features'].split(',')]

    champion_config_path = os.path.join(results_dir, 'champion_config.yaml')
    with open(champion_config_path, 'w') as f:
        yaml.dump(champion_config, f, default_flow_style=False, sort_keys=False, indent=2)
    print(f"Configuração do modelo campeão salva em: {champion_config_path}")

    # Gera e salva o gráfico final
    final_plot_path = os.path.join(plots_dir, f'full_optimization_summary_{timestamp}.png')
    plot_full_optimization_results(final_results_df, final_plot_path)
    print("#"*80)

if __name__ == "__main__":
    main()
