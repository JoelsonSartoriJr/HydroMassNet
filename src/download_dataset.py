import pandas as pd
import numpy as np
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch
from astropy import units as u
import os
import sys
import tempfile

DATA_DIR = 'data'
RAW_DATA_PATH = os.path.join(DATA_DIR, 'alfalfa_raw_full.csv')
FULL_DATASET_PATH = os.path.join(DATA_DIR, 'hydromassnet_full_dataset_all_columns.csv')
VIZIER_CATALOG = "J/AJ/160/271"
TARGET_COLUMN = 'logMHI'

def find_coord_columns(df):
    """
    Encontra dinamicamente os nomes das colunas de RA e Dec.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados do catálogo.

    Returns
    -------
    tuple
        Nomes das colunas de RA e Dec.
    """
    ra_possible_names = ['RAdeg', 'ra', '_RA', 'RAJ2000']
    dec_possible_names = ['DEdeg', 'dec', '_DE', 'DEJ2000']

    ra_col = next((name for name in ra_possible_names if name in df.columns), None)
    dec_col = next((name for name in dec_possible_names if name in df.columns), None)

    if ra_col and dec_col:
        print(f"Colunas de coordenadas identificadas como: '{ra_col}' e '{dec_col}'")
    return ra_col, dec_col

def fetch_base_catalog(cache_path=RAW_DATA_PATH):
    """
    Baixa e mescla as tabelas do catálogo ALFALFA-SDSS de forma robusta.

    Parameters
    ----------
    cache_path : str, optional
        Caminho para o cache.

    Returns
    -------
    pd.DataFrame
        DataFrame com dados do catálogo.
    """
    if os.path.exists(cache_path):
        print(f"--- Usando dados base cacheados de '{cache_path}' ---")
        df_cache = pd.read_csv(cache_path)
        if TARGET_COLUMN in df_cache.columns:
            return df_cache
        print(f"AVISO: Cache está incompleto (falta '{TARGET_COLUMN}'). Removendo para baixar novamente.", file=sys.stderr)
        os.remove(cache_path)

    print(f"--- Etapa 1: Baixando o catálogo completo de {VIZIER_CATALOG} via VizieR ---")
    try:
        vizier = Vizier(catalog=VIZIER_CATALOG, columns=['**'], row_limit=-1)
        tables = vizier.get_catalogs(VIZIER_CATALOG)
        if len(tables) < 2:
            raise ValueError("VizieR não retornou as duas tabelas esperadas do catálogo.")

        obs_props = tables[0].to_pandas()
        der_props = tables[1].to_pandas()
        print(f"Tabela 1 (observações) carregada com {len(obs_props)} linhas.")
        print(f"Tabela 2 (derivados) carregada com {len(der_props)} linhas.")

        if TARGET_COLUMN not in der_props.columns:
            print(f"ERRO: A coluna alvo '{TARGET_COLUMN}' não foi encontrada na segunda tabela do VizieR.", file=sys.stderr)
            print(f"Colunas disponíveis na tabela 2: {der_props.columns.tolist()}", file=sys.stderr)
            sys.exit(1)

        common_cols = list(set(obs_props.columns) & set(der_props.columns) - {'AGC'})
        obs_props.rename(columns={col: f"{col}_obs" for col in common_cols}, inplace=True)

        print(f"Mesclando as duas tabelas pela coluna 'AGC'...")
        alfalfa_full = pd.merge(obs_props, der_props, on='AGC', how='outer')

        ra_col_name, dec_col_name = find_coord_columns(alfalfa_full)
        if not ra_col_name or not dec_col_name:
             raise KeyError("Não foi possível identificar as colunas de coordenadas no dataframe final.")

        alfalfa_full.rename(columns={ra_col_name: 'ra', dec_col_name: 'dec'}, inplace=True)
        alfalfa_full.dropna(subset=['ra', 'dec', TARGET_COLUMN], inplace=True)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        alfalfa_full.to_csv(cache_path, index=False)
        print(f"Tabela base completa com {len(alfalfa_full)} galáxias salva em '{cache_path}'.")
        return alfalfa_full

    except Exception as e:
        print(f"ERRO CRÍTICO na Etapa 1: {e}", file=sys.stderr)
        sys.exit(1)

def query_and_merge_external_data(base_df):
    """
    Realiza cross-match e mescla dados de catálogos externos.

    Parameters
    ----------
    base_df : pd.DataFrame
        DataFrame base para o cross-match.

    Returns
    -------
    pd.DataFrame
        DataFrame com dados externos mesclados.
    """
    print("\n--- Etapa 2: Realizando cross-match com catálogos externos ---")

    coords_df = base_df[['ra', 'dec']].copy()

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as temp_f:
        coords_df.to_csv(temp_f.name, index=False)
        temp_filename = temp_f.name

    try:
        catalogs_to_match = [
            "vizier:V/154/sdss16", "vizier:II/246/out",
            "vizier:II/328/allwise", "vizier:II/335/galex_ais"
        ]

        xmatch = XMatch()
        matched_table = xmatch.query(
            cat1=open(temp_filename), cat2=catalogs_to_match,
            max_distance=10 * u.arcsec, colRA1='ra', colDec1='dec'
        )

        matched_df = matched_table.to_pandas()
        print(f"Cross-match retornou {len(matched_df)} correspondências.")

        matched_df = matched_df.sort_values('angDist').drop_duplicates(['ra', 'dec'])
        enriched_df = pd.merge(base_df, matched_df, on=['ra', 'dec'], how='left')
        print(f"Dataset enriquecido possui {len(enriched_df.columns)} colunas.")
        return enriched_df

    except Exception as e:
        print(f"AVISO: Falha no cross-match: {e}. Prosseguindo apenas com os dados base.", file=sys.stderr)
        return base_df
    finally:
        os.remove(temp_filename)

def finalize_dataset(df):
    """
    Limpa o dataset final e prepara para o uso.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser finalizado.

    Returns
    -------
    pd.DataFrame
        DataFrame final limpo.
    """
    print("\n--- Etapa 3: Finalizando o dataset ---")

    print(f"Shape antes da limpeza final: {df.shape}")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='ignore')

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    print(f"Valores numéricos ausentes preenchidos com a mediana.")
    return df

def main():
    """
    Orquestra o pipeline completo de criação do dataset.
    """
    base_catalog = fetch_base_catalog()
    enriched_catalog = query_and_merge_external_data(base_catalog)
    final_dataset = finalize_dataset(enriched_catalog)

    final_dataset.to_csv(FULL_DATASET_PATH, index=False)
    print(f"\n✅ SUCESSO! Dataset completo salvo em: '{FULL_DATASET_PATH}'")
    print(f"O dataset final contém {len(final_dataset)} galáxias e {len(final_dataset.columns)} colunas.")

if __name__ == '__main__':
    Vizier.TIMEOUT = 120
    main()
