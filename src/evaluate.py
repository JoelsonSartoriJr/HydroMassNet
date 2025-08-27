import argparse
import yaml
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data import DataHandler
from src.models.bayesian_dense_regression import BayesianDenseRegression
from src.models.bayesian_density_network import BayesianDensityNetwork
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tqdm import tqdm

def create_vanilla_model(layer_dims, input_shape, dropout_rate):
    """
    Cria um modelo sequencial simples usando a API funcional do Keras.

    Parameters
    ----------
    layer_dims : list
        Lista de inteiros com as dimensões de cada camada.
    input_shape : tuple
        O formato de entrada do modelo.
    dropout_rate : float
        Taxa de dropout para as camadas ocultas.

    Returns
    -------
    tf.keras.Model
        O modelo Keras.
    """
    layers = [Input(shape=input_shape, name="input_layer")]
    for units in layer_dims[:-1]:
        layers.extend([Dense(units, activation='relu'), Dropout(dropout_rate)])
    layers.append(Dense(layer_dims[-1]))
    return Sequential(layers, name='vanilla')

def main(args):
    """
    Função principal para avaliar um modelo.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_handler = DataHandler(config)
    _, _, _, _, x_test, y_test, features = data_handler.get_full_dataset_and_splits()

    model_config = config['models'][args.model]
    model_path = os.path.join(config['paths']['saved_models'], model_config['save_name'])
    input_shape = (len(features),)

    if args.model == 'bnn':
        model = BayesianDenseRegression(model_config['layers'])
        model.build((None, *input_shape))
    elif args.model == 'dbnn':
        model = BayesianDensityNetwork(model_config['core_layers'], model_config['head_layers'])
        model.build((None, *input_shape))
    elif args.model == 'vanilla':
        layers = model_config['layers'][1:]
        model = create_vanilla_model(layers, input_shape, model_config['dropout'])
    else:
        raise ValueError(f"Modelo '{args.model}' não é suportado.")

    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.load_weights(model_path)

    print(f"Avaliando o modelo {args.model.upper()}...")

    if args.model in ['bnn', 'dbnn']:
        n_samples = 100
        predictions_samples = []
        for _ in tqdm(range(n_samples), desc=f"Amostrando {args.model.upper()}"):
            y_pred_dist = model(x_test, sampling=True)
            predictions_samples.append(y_pred_dist.mean().numpy())

        predictions_samples = np.array(predictions_samples)
        y_pred_mean = predictions_samples.mean(axis=0)
        y_pred_std = predictions_samples.std(axis=0)

        results_df = pd.DataFrame({
            'y_true': y_test.flatten(),
            'y_pred_mean': y_pred_mean.flatten(),
            'y_pred_std': y_pred_std.flatten()
        })
    else:
        y_pred = model.predict(x_test)
        results_df = pd.DataFrame({
            'y_true': y_test.flatten(),
            'y_pred_mean': y_pred.flatten(),
            'y_pred_std': np.zeros_like(y_pred.flatten())
        })

    output_path = os.path.join(config['paths']['saved_models'], f'{args.model}_predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Predições (com incerteza) salvas em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliador de Modelos para HydroMassNet")
    parser.add_argument('--model', type=str, required=True, choices=['bnn', 'dbnn', 'vanilla'], help='Qual modelo avaliar.')
    args = parser.parse_args()
    main(args)
