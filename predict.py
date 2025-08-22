import argparse
import yaml
import os
import joblib
import numpy as np
import tensorflow as tf

from src.models.bayesian_dense_regression import BayesianDenseRegression
from src.models.bayesian_density_network import BayesianDensityNetwork

def main(args):
    # Carregar configurações
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Carregar scalers
    scaler_x_path = os.path.join(config['paths']['saved_models'], 'scaler_x.pkl')
    scaler_y_path = os.path.join(config['paths']['saved_models'], 'scaler_y.pkl')
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)

    # Preprocessar entrada
    input_values = np.array(args.input_values).reshape(1, -1)
    if input_values.shape[1] != len(config['data_processing']['features']):
        raise ValueError(f"Esperava {len(config['data_processing']['features'])} valores de entrada, mas recebeu {input_values.shape[1]}")

    input_scaled = scaler_x.transform(input_values)

    # Instanciar e carregar modelo
    model_config = config['models'][args.model]
    input_shape = (input_scaled.shape[1],)

    if args.model == 'bnn':
        model = BayesianDenseRegression(model_config['layers'])
        model.build(input_shape)
    elif args.model == 'dbnn':
        model = BayesianDensityNetwork(model_config['core_layers'], model_config['head_layers'])
        model.build(input_shape)
    else:
         raise ValueError(f"Modelo '{args.model}' não é suportado.")

    model_path = os.path.join(config['paths']['saved_models'], model_config['save_name'])
    model.load_weights(model_path)
    print(f"Pesos do modelo carregados de {model_path}")

    # Fazer predições
    num_samples = 100
    predictions_scaled = np.array([model(input_scaled, training=True).numpy().flatten() for _ in range(num_samples)])

    # Inverter a transformação para a escala original
    # predictions_scaled tem shape (num_samples, 2) -> [loc, std]
    loc_preds_scaled = predictions_scaled[:, 0].reshape(-1, 1)

    predictions_real = scaler_y.inverse_transform(loc_preds_scaled)

    mean_pred = np.mean(predictions_real)
    std_pred = np.std(predictions_real)

    print("\n--- Resultados da Predição ---")
    print(f"Modelo: {args.model.upper()}")
    print(f"Média da Previsão (logMH): {mean_pred:.4f}")
    print(f"Desvio Padrão da Previsão (incerteza): {std_pred:.4f}")
    print("----------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predição com Modelos HydroMassNet')
    parser.add_argument('--model', type=str, required=True, choices=['bnn', 'dbnn'], help='Qual modelo usar para predição.')
    parser.add_argument('--input_values', type=float, nargs='+', required=True, help='Valores de entrada para as features (13 valores).')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Caminho para o arquivo de configuração.')
    args = parser.parse_args()
    main(args)
