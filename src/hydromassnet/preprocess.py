# file: ./src/hydromassnet/preprocess.py
import pandas as pd
import numpy as np
import os

def clean_and_feature_engineer(config: dict):
    """
    Carrega os dados brutos, limpa, cria features e salva o dataset processado.
    """
    print("--- Iniciando pré-processamento e engenharia de features ---")

    paths = config['paths']
    df = pd.read_csv(paths['raw_data'])

    df.replace([-99.99, -99.0], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            # Correção para evitar o FutureWarning do Pandas
            df[col] = df[col].fillna(median_val)

    if 'iMAG' in df.columns and 'b/a' in df.columns:
        df['surface_brightness_proxy'] = df['iMAG'] + 2.5 * np.log1p(df['b/a'])

    all_features = set()
    for model_cfg in config['models'].values():
        all_features.update(model_cfg['features'])

    target_column = config['target_column']
    all_features.add(target_column)

    final_cols = [col for col in all_features if col in df.columns]
    df_processed = df[final_cols].copy()
    df_processed.dropna(inplace=True)

    processed_path = paths['processed_data']
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_processed.to_csv(processed_path, index=False)

    print(f"\nDataset processado com {len(df_processed.columns)-1} features.")
    print(f"Salvo em: {processed_path}")
    print("--- Pré-processamento concluído ---")
