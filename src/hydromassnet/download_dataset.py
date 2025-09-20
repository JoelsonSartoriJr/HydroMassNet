# file: ./src/hydromassnet/download_dataset.py
import pandas as pd
import numpy as np
from astroquery.vizier import Vizier
import os
import sys
import yaml

def fetch_base_catalog(config):
    """Baixa e mescla as tabelas do catálogo ALFALFA-SDSS de forma robusta."""
    raw_data_path = config['paths'].get('raw_data_temp', 'data/alfalfa_raw_full.csv')
    target_column = config['target_column']
    vizier_catalog = "J/AJ/160/271"

    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)

    if os.path.exists(raw_data_path):
        print(f"--- Usando dados base cacheados de '{raw_data_path}' ---")
        return pd.read_csv(raw_data_path)

    print(f"--- Baixando o catálogo de {vizier_catalog} via VizieR ---")
    try:
        vizier = Vizier(catalog=vizier_catalog, columns=['**'], row_limit=-1)
        tables = vizier.get_catalogs(vizier_catalog)
        if len(tables) < 2:
            raise ValueError("VizieR não retornou as duas tabelas esperadas.")

        obs_props, der_props = tables[0].to_pandas(), tables[1].to_pandas()

        if target_column not in der_props.columns:
            raise ValueError(f"Coluna alvo '{target_column}' não encontrada.")

        common_cols = list(set(obs_props.columns) & set(der_props.columns) - {'AGC'})
        obs_props.rename(columns={col: f"{col}_obs" for col in common_cols}, inplace=True)

        alfalfa_full = pd.merge(obs_props, der_props, on='AGC', how='outer')
        alfalfa_full.dropna(subset=[target_column], inplace=True)

        alfalfa_full.to_csv(raw_data_path, index=False)
        print(f"Tabela base salva em '{raw_data_path}'.")
        return alfalfa_full

    except Exception as e:
        print(f"ERRO CRÍTICO ao baixar dados: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Orquestra o pipeline completo de criação do dataset."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset = fetch_base_catalog(config)

    final_path = config['paths']['raw_data']
    dataset.to_csv(final_path, index=False)
    print(f"\n✅ SUCESSO! Dataset completo salvo em: '{final_path}'")
    print(f"O dataset final contém {len(dataset)} galáxias e {len(dataset.columns)} colunas.")

if __name__ == '__main__':
    Vizier.TIMEOUT = 120
    main()
