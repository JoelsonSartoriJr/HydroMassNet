import pandas as pd
import numpy as np
import os
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from astropy import units as u

def get_vizier_data(catalog_id="J/AJ/160/271"):
    """
    Baixa as tabelas de um catálogo do VizieR.
    """
    print(f"--- Etapa 1: Baixando o catálogo completo de {catalog_id} via VizieR ---")
    v = Vizier(columns=['*'])
    v.ROW_LIMIT = -1
    table_list = v.get_catalogs(catalog_id)

    table1 = table_list[0].to_pandas()
    table2 = table_list[1].to_pandas()

    print(f"Tabela 1 (observações) carregada com {len(table1)} linhas.")
    print(f"Tabela 2 (derivados) carregada com {len(table2)} linhas.")

    print("Mesclando as duas tabelas pela coluna 'AGC'...")
    merged_df = pd.merge(table1, table2, on='AGC', how='outer')

    coord_cols = ['RAJ2000', 'DEJ2000']
    print(f"Colunas de coordenadas identificadas como: {coord_cols}")

    raw_path = 'data/alfalfa_raw_full.csv'
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    merged_df.to_csv(raw_path, index=False)
    print(f"Tabela base completa com {len(merged_df)} galáxias salva em '{raw_path}'.\n")
    return merged_df

def cross_match_with_external_catalogs(df):
    """
    Realiza o cross-match com outros catálogos astronômicos (ex: 2MASS).
    """
    print("--- Etapa 2: Realizando cross-match com catálogos externos ---")
    try:
        coords = SkyCoord(ra=df['RAJ2000'], dec=df['DEJ2000'], unit=(u.deg, u.deg), frame='icrs')

        # Cross-match com 2MASS
        result_tables = Vizier.query_region(coords, radius=10 * u.arcsec, catalog='II/246/out')

        # --- CORREÇÃO: Pega a primeira tabela da lista retornada ---
        if result_tables:
            cross_matched_data = result_tables[0].to_pandas()
            # Renomeia colunas para evitar conflitos antes do merge
            cross_matched_data.rename(columns={'RAJ2000': 'RAJ2000_2MASS', 'DEJ2000': 'DEJ2000_2MASS'}, inplace=True)
            df = pd.merge(df, cross_matched_data, left_on='AGC', right_on='_2MASX', how='left')
            print("Cross-match com 2MASS realizado com sucesso.")
        else:
            print("Nenhum resultado encontrado no cross-match com 2MASS.")

    except Exception as e:
        # A falha agora é mais informativa, mas o script continua
        print(f"AVISO: Falha no cross-match: {e}. Prosseguindo apenas com os dados base.")

    print("")
    return df

def finalize_dataset(df):
    """
    Limpa e finaliza o dataset, tratando valores ausentes e convertendo tipos.
    """
    print("--- Etapa 3: Finalizando o dataset ---")
    print(f"Shape antes da limpeza final: {df.shape}")

    # Substitui valores sentinela por NaN
    df.replace([-99.99, -99.0], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        # --- CORREÇÃO 1: Conversão de tipo mais segura e moderna ---
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- CORREÇÃO 2: Preenchimento de nulos de forma segura e recomendada ---
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    print("Valores numéricos ausentes preenchidos com a mediana.")

    final_path = 'data/hydromassnet_full_dataset_all_columns.csv'
    df.to_csv(final_path, index=False)

    print(f"\n✅ SUCESSO! Dataset completo salvo em: '{final_path}'")
    print(f"O dataset final contém {df.shape[0]} galáxias e {df.shape[1]} colunas.")

if __name__ == "__main__":
    base_data = get_vizier_data()
    data_with_crossmatch = cross_match_with_external_catalogs(base_data)
    finalize_dataset(data_with_crossmatch)
