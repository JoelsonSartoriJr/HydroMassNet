import pandas as pd
from astroquery.vizier import Vizier
from astropy import coordinates as coords
import astropy.units as u
import numpy as np
import os
import requests
from io import StringIO

def query_sdss_api(sql_query):
    """
    Envia uma consulta SQL diretamente para a API do SDSS SciServer,
    conforme recomendado pela documentação DR19.
    """
    # Endpoint da API do SciServer CasJobs
    url = "https://skyserver.sdss.org/casjobs/RestAPI/contexts/dr17/query"

    # Parâmetros para a requisição
    params = {
        'query': sql_query,
        'format': 'csv' # Pedimos o resultado em formato CSV
    }

    print("Enviando consulta para a API do SDSS SciServer...")
    try:
        response = requests.get(url, params=params, timeout=300) # Timeout de 5 minutos

        # Verifica se a requisição foi bem-sucedida
        response.raise_for_status()

        # A resposta vem como texto, precisamos convertê-la em um DataFrame
        # Pulamos a primeira linha que é o cabeçalho do resultado da query
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        return df

    except requests.exceptions.HTTPError as http_err:
        print(f"Erro HTTP ao contatar o servidor SDSS: {http_err}")
        if response.status_code == 503:
            print(">>> O servidor retornou '503 Service Unavailable'.")
            print(">>> Isso significa que o serviço do SDSS está temporariamente fora do ar ou sobrecarregado.")
            print(">>> Por favor, tente executar o script novamente mais tarde.")
        else:
            print(f"Conteúdo da resposta do servidor: {response.text}")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante a consulta ao SDSS: {e}")
        return None


def build_complete_dataset(save_path='data/master_dataset.csv', galaxy_sample_limit=100000):
    """
    Constrói um dataset mestre usando a API do SciServer e cruzando com
    outros levantamentos para obter o maior número de objetos possível.
    """

    print(f"--- Etapa 1: Consultando o SDSS para uma amostra de até {galaxy_sample_limit} galáxias via API ---")

    sdss_query = f"""
    SELECT TOP {galaxy_sample_limit}
        p.objID, p.ra, p.dec,
        p.modelMag_u as umag, p.modelMag_g as gmag,
        p.modelMag_r as rmag, p.modelMag_i as imag,
        s.z,
        p.expAB_r
    FROM
        PhotoObj AS p
    JOIN
        SpecObj AS s ON s.bestObjID = p.objID
    WHERE
        p.type = 3 AND s.zStatus = 4 AND s.zWarning = 0 AND s.sciencePrimary = 1
    """

    sdss_df = query_sdss_api(sdss_query)

    if sdss_df is None:
        print("A construção do dataset foi abortada devido a uma falha na consulta ao SDSS.")
        return

    light_speed_kms = 299792.458
    sdss_df['Vhelio'] = sdss_df['z'] * light_speed_kms
    print(f"SDSS retornou {len(sdss_df)} galáxias com dados espectroscópicos.")

    # O restante do script continua como antes...
    print("\n--- Etapa 2: Cruzando as galáxias do SDSS com o catálogo AllWISE ---")
    # ... (o código das etapas 2, 3 e 4 permanece o mesmo)
    Vizier.ROW_LIMIT = -1
    wise_catalog = "II/328/allwise"
    sdss_coords = coords.SkyCoord(ra=sdss_df['ra'].values*u.degree, dec=sdss_df['dec'].values*u.degree)

    try:
        wise_results = Vizier.query_region(sdss_coords, radius=3*u.arcsec, catalog=wise_catalog)
        if not wise_results:
            print("O cross-match com o WISE não retornou resultados. Continuando sem dados WISE.")
            merged_df = sdss_df
        else:
            wise_df = wise_results[0].to_pandas()
            print(f"WISE retornou {len(wise_df)} contrapartes.")
            sdss_df['match_idx'] = np.arange(len(sdss_df))
            merged_df = pd.merge(sdss_df, wise_df, left_on='match_idx', right_on='_q', how='left')

    except Exception as e:
        print(f"Falha no cross-match com o WISE: {e}")
        merged_df = sdss_df

    print("\n--- Etapa 3: Cruzando os resultados com o catálogo ALFALFA para obter a massa de HI ---")

    alfalfa_catalog = "J/AJ/156/256/alfalfa"
    merged_coords = coords.SkyCoord(ra=merged_df['ra'].values*u.degree, dec=merged_df['dec'].values*u.degree)

    try:
        alfalfa_results = Vizier.query_region(merged_coords, radius=1*u.arcmin, catalog=alfalfa_catalog)
        if not alfalfa_results:
            print("O cross-match com o ALFALFA não retornou resultados. Continuando sem dados ALFALFA.")
            final_df = merged_df
        else:
            alfalfa_df = alfalfa_results[0].to_pandas()
            print(f"ALFALFA retornou {len(alfalfa_df)} contrapartes.")
            merged_df['match_idx_2'] = np.arange(len(merged_df))
            final_df = pd.merge(merged_df, alfalfa_df, left_on='match_idx_2', right_on='_q', how='left')

    except Exception as e:
        print(f"Falha no cross-match com o ALFALFA: {e}")
        final_df = merged_df

    print("\n--- Etapa 4: Calculando features finais e limpando o dataset ---")

    if 'W1mag' in final_df.columns and 'W2mag' in final_df.columns:
        final_df['logMstarMcGaugh'] = -2.48 * (final_df['W1mag'] - final_df['W2mag']) + 0.95 * (4.64 - final_df['W1mag'])
    if 'W4mag' in final_df.columns:
        final_df['logSFR22'] = 1.05 * (7.6 - 0.4 * final_df['W4mag']) - 7.55

    if 'logMHI' in final_df.columns:
        final_df.rename(columns={'logMHI': 'logMH'}, inplace=True)

    final_df['g-r'] = final_df['gmag'] - final_df['rmag']
    final_df['u-r'] = final_df['umag'] - final_df['rmag']
    final_df['gmi_corr'] = final_df['gmag'] - final_df['imag']

    final_columns = [
        'umag', 'gmag', 'rmag', 'imag', 'g-r', 'u-r', 'Vhelio',
        'expAB_r', 'gmi_corr', 'logMstarMcGaugh', 'logSFR22', 'logMH'
    ]

    available_features = [col for col in final_columns if col in final_df.columns]
    print(f"Features finais que serão incluídas ({len(available_features)}): {available_features}")

    master_df = final_df[available_features]

    print(f"\nShape antes da limpeza de NaNs: {master_df.shape}")
    master_df.dropna(inplace=True)
    print(f"Shape após a limpeza: {master_df.shape}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    master_df.to_csv(save_path, index=False)

    print(f"\nSUCESSO! Dataset mestre construído e salvo em: '{save_path}'")


if __name__ == '__main__':
    Vizier.TIMEOUT = 120
    build_complete_dataset()
