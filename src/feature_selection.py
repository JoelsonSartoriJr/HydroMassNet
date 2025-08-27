import pandas as pd
import yaml
import matplotlib
# CORREÇÃO: Define o backend não interativo para matplotlib
matplotlib.use('agg')
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os

# CORREÇÃO: Alterado para importação relativa
from .data import DataHandler

def run_feature_selection(config: dict):
    """
    Executa a seleção de features usando LightGBM e SHAP.
    """
    print("\n--- Iniciando Seleção de Features com LightGBM e SHAP ---")

    # 1. Carrega os dados
    data_handler = DataHandler(config=config)
    x_train, y_train, x_val, y_val, _, _, feature_names = data_handler.get_full_dataset_and_splits()

    # 2. Treina o modelo LightGBM
    lgb_train = lgb.Dataset(x_train, y_train, feature_name=feature_names)
    lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train, feature_name=feature_names)

    params = config['feature_selection']['lgbm_params']

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[lgb.early_stopping(10, verbose=False)]
    )

    # 3. Calcula os valores SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_val)

    # 4. Processa e seleciona as features
    mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=feature_names)
    mean_abs_shap = mean_abs_shap.sort_values(ascending=False)

    shap_threshold = config['feature_selection']['shap_threshold']
    selected_features = mean_abs_shap[mean_abs_shap > shap_threshold].index.tolist()

    print(f"\n--- {len(selected_features)} features selecionadas com SHAP > {shap_threshold} ---")
    print(selected_features)

    # 5. Salva o gráfico de importância das features
    reports_dir = config['paths']['reports']
    os.makedirs(reports_dir, exist_ok=True)

    plt.figure(figsize=(10, len(feature_names) / 2))
    shap.summary_plot(shap_values, x_val, feature_names=feature_names, show=False, plot_type="bar")
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, 'feature_importance_shap.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"\n--- Gráfico de importância SHAP salvo em: {plot_path} ---")

    return selected_features

if __name__ == '__main__':
    with open('config/config.yaml', 'r') as f:
        main_config = yaml.safe_load(f)
    run_feature_selection(config=main_config)
