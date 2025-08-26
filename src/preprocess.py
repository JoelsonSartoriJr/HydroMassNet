import pandas as pd
import numpy as np
import yaml
import os

def clean_and_feature_engineer(config: dict):
    """
    Carrega os dados brutos, limpa, cria features e salva o dataset processado.
    """
    print("--- Iniciando pré-processamento e engenharia de features ---")

    df = pd.read_csv(config['paths']['raw_data'])

    # Substitui valores sentinela por NaN
    df.replace([-99.99, -99.0], np.nan, inplace=True)

    # Preenche valores ausentes com a mediana da coluna
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # --- CORREÇÃO: Usa a fórmula correta com a coluna 'b/a' que existe no dataset ---
    # np.log1p(x) é mais estável numericamente que np.log(1+x)
    df['surface_brightness_proxy'] = df['iMAG'] + 2.5 * np.log1p(df['b/a'])

    # Garante que todas as colunas necessárias existem antes de selecionar
    features_to_keep = config['data_processing']['features']
    target_column = config['target_column']

    final_cols = [col for col in features_to_keep + [target_column] if col in df.columns]
    df_processed = df[final_cols].copy()
    df_processed.dropna(inplace=True)

    processed_path = config['paths']['processed_data']
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False)

    print(f"\nDataset processado com {len(df_processed.columns)-1} features.")
    print(f"Salvo em: {processed_path}")
    print("--- Pré-processamento concluído ---")

if __name__ == "__main__":
    # Carrega a configuração a partir do diretório raiz do projeto
    with open('config/config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    clean_and_feature_engineer(config_data)
