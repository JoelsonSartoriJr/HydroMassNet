# file: ./evaluate.py
import argparse
import yaml
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from src.hydromassnet.data import DataHandler
from src.hydromassnet.models import model_factory

def main(model_name: str):
    """
    Função principal para avaliar um modelo treinado.

    Parameters
    ----------
    model_name : str
        O nome do modelo a ser avaliado.
    """
    print(f"--- Iniciando Avaliação para o Modelo: {model_name.upper()} ---")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['models'][model_name]
    paths_cfg = config['paths']

    print("--- Carregando e preparando dados de teste ---")
    data_handler = DataHandler(config, feature_override=model_cfg['features'])
    _, _, _, _, x_test, y_test, features = data_handler.get_full_dataset_and_splits()

    input_shape = len(features)
    model = model_factory(model_cfg, input_shape, 1)

    model_path = os.path.join(paths_cfg['models'], f'{model_name}.weights.h5')
    print(f"--- Carregando pesos do modelo de '{model_path}' ---")
    model.load_weights(model_path)

    if model_cfg['type'] in ['bnn', 'dbnn']:
        n_samples = config['training']['evaluation_samples']
        print(f"--- Realizando {n_samples} passagens para estimar a incerteza ---")

        # Since we're using regular Dense layers with dropout, we enable training=True for uncertainty estimation
        y_pred_samples = [model(x_test, training=True).numpy() for _ in tqdm(range(n_samples))]
        y_pred_samples = np.array(y_pred_samples)
        y_pred_mean = y_pred_samples.mean(axis=0)
        y_pred_std = y_pred_samples.std(axis=0)
    else:
        print("--- Realizando predições no conjunto de teste ---")
        y_pred_mean = model.predict(x_test)
        y_pred_std = np.zeros_like(y_pred_mean)

    print("--- Desescalonando predições para a escala original ---")
    y_true_real = data_handler.scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_pred_mean_real = data_handler.scaler_y.inverse_transform(y_pred_mean.reshape(-1, 1))

    scale_factor = data_handler.scaler_y.scale_[0]
    y_pred_std_real = y_pred_std / scale_factor

    results_df = pd.DataFrame({
        'y_true': y_true_real.flatten(),
        'y_pred_mean': y_pred_mean_real.flatten(),
        'y_pred_std': y_pred_std_real.flatten()
    })

    output_path = os.path.join(paths_cfg['results'], f'{model_name}_predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"--- Avaliação concluída. Predições salvas em: {output_path} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script de Avaliação HydroMassNet')
    parser.add_argument('--model', type=str, required=True,
                        help='Nome do modelo a ser avaliado (ex: bnn, dbnn, vanilla).')
    args = parser.parse_args()
    main(args.model)
