# file: ./predict.py
import argparse
import yaml
import os
import joblib
import numpy as np
import tensorflow as tf
from src.hydromassnet.models import model_factory

def main(args):
    """
    Faz predições com incerteza para um conjunto de dados de entrada.
    """
    print(f"--- Iniciando Predição com o Modelo: {args.model.upper()} ---")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['models'][args.model]
    paths_cfg = config['paths']

    scaler_x_path = os.path.join(paths_cfg['results'], 'scaler_x.pkl')
    scaler_y_path = os.path.join(paths_cfg['results'], 'scaler_y.pkl')

    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        raise FileNotFoundError("Scalers 'scaler_x.pkl' ou 'scaler_y.pkl' não encontrados. "
                                "Execute o treinamento primeiro.")

    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    features = model_cfg['features']
    input_values = np.array(args.input_values).reshape(1, -1)

    if input_values.shape[1] != len(features):
        raise ValueError(f"Esperado {len(features)} valores de entrada para as features, "
                         f"mas {input_values.shape[1]} foram fornecidos.")

    input_scaled = scaler_x.transform(input_values)

    input_shape = len(features)
    model = model_factory(model_cfg, input_shape, 1)

    model_path = os.path.join(paths_cfg['models'], f'{args.model}.weights.h5')
    print(f"--- Carregando pesos do modelo de '{model_path}' ---")
    model.load_weights(model_path)

    if model_cfg['type'] in ['bnn', 'dbnn']:
        n_samples = config['training']['evaluation_samples']
        # Since we're using regular Dense layers with dropout, we enable training=True for uncertainty estimation
        y_pred_samples = [model(input_scaled, training=True).numpy() for _ in range(n_samples)]
        y_pred_samples = np.array(y_pred_samples)
        y_pred_mean_scaled = y_pred_samples.mean(axis=0)
        y_pred_std_scaled = y_pred_samples.std(axis=0)
    else: # Vanilla
        y_pred_mean_scaled = model.predict(input_scaled)
        y_pred_std_scaled = np.zeros_like(y_pred_mean_scaled)

    mean_pred_real = scaler_y.inverse_transform(y_pred_mean_scaled.reshape(1, -1))
    std_pred_real = y_pred_std_scaled / scaler_y.scale_[0]

    print("\n--- Resultado da Predição ---")
    print(f"Modelo: {args.model.upper()}")
    print(f"Média da Previsão (logMHI): {mean_pred_real[0][0]:.4f}")
    print(f"Incerteza (desvio padrão): {std_pred_real[0][0]:.4f}")
    print("-----------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predição com Modelos HydroMassNet')
    parser.add_argument('--model', type=str, required=True, choices=['bnn', 'dbnn', 'vanilla'],
                        help='Qual modelo treinado usar para a predição.')
    parser.add_argument('--input_values', type=float, nargs='+', required=True,
                        help='Valores de entrada para as features, na ordem definida no config.yaml.')
    args = parser.parse_args()
    main(args)
