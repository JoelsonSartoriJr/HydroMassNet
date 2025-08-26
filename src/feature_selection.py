import pandas as pd
import yaml
import itertools
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os

def run_feature_selection(config: dict) -> str:
    """
    Executa a seleção de features usando LightGBM e busca por combinações ideais.

    Parameters
    ----------
    config : dict
        Dicionário de configurações do projeto.

    Returns
    -------
    str
        O caminho do arquivo com os resultados da seleção.
    """
    print("--- Iniciando o processo de Seleção de Features ---")

    data = pd.read_csv(config['paths']['processed_data'])
    candidate_features = config['feature_selection']['candidates']
    target = config['target_column']
    candidate_features = [f for f in candidate_features if f in data.columns]
    print(f"Testando combinações de {len(candidate_features)} features candidatas.")

    X = data[candidate_features]
    y = data[target]

    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=candidate_features)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y,
        test_size=config['data_processing']['val_split'],
        random_state=config['seed']
    )

    results = []
    min_features = config['feature_selection']['min_features']

    all_combinations = []
    for i in range(min_features, len(candidate_features) + 1):
        all_combinations.extend(itertools.combinations(candidate_features, i))

    print(f"Total de combinações a serem testadas: {len(all_combinations)}")

    for combo in tqdm(all_combinations, desc="Testando Combinações de Features"):
        combo_list = list(combo)

        lgbm = lgb.LGBMRegressor(random_state=config['seed'], n_estimators=200, verbose=-1)
        lgbm.fit(X_train[combo_list], y_train)

        y_pred = lgbm.predict(X_val[combo_list])
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        results.append({
            'num_features': len(combo_list),
            'features': ", ".join(combo_list),
            'rmse': rmse,
            'r2': r2
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=['r2', 'rmse'], ascending=[False, True])

    results_dir = config['paths']['search_results']
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = os.path.join(results_dir, f'feature_selection_results_{timestamp}.csv')
    results_df.to_csv(results_filename, index=False)

    print("\n" + "="*80)
    print("--- Seleção de Features Concluída ---")
    print(f"Resultados salvos em: {results_filename}")
    print("\n--- 5 Melhores Combinações de Features Encontradas ---")
    print(results_df.head(5).to_string())
    print("="*80)

    return results_filename

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    run_feature_selection(config_data)
