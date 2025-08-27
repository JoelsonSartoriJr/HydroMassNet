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
    Cria um modelo sequencial de forma robusta.

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
    model = Sequential(name='vanilla')
    model.add(Input(shape=input_shape))
    for units in layer_dims[:-1]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(layer_dims[-1]))
    return model

def main(args):
    """
    Função principal para avaliar os modelos campeões.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['champion_models'][args.model]

    data_handler = DataHandler(config, feature_override=model_cfg['features'])
    _, _, _, _, x_test, y_test, features = data_handler.get_full_dataset_and_splits()

    input_shape = (len(features),)

    if args.model == 'bnn':
        model = BayesianDenseRegression(model_cfg['layers'])
        model.build((None, *input_shape))
    elif args.model == 'dbnn':
        model = BayesianDensityNetwork(model_cfg['core_layers'], model_cfg['head_layers'])
        model.build((None, *input_shape))
    elif args.model == 'vanilla':
        layers_to_create = model_cfg['layers'][1:]
        model = create_vanilla_model(layers_to_create, input_shape, model_cfg['dropout'])

    model.compile(optimizer=tf.keras.optimizers.Adam())
    model_path = os.path.join(config['paths']['saved_models'], model_cfg['save_name'])
    model.load_weights(model_path)

    print(f"--- Avaliando o modelo campeão '{args.model}' ---")

    if args.model in ['bnn', 'dbnn']:
        n_samples = 100
        samples = [model(x_test, sampling=False).mean().numpy() for _ in tqdm(range(n_samples), desc="Amostragem")]
        samples = np.array(samples)
        y_pred_mean = samples.mean(axis=0)
        y_pred_std = samples.std(axis=0)
    else:
        y_pred_mean = model.predict(x_test)
        y_pred_std = np.zeros_like(y_pred_mean)

    results_df = pd.DataFrame({
        'y_true': y_test.flatten(),
        'y_pred_mean': y_pred_mean.flatten(),
        'y_pred_std': y_pred_std.flatten()
    })

    output_path = os.path.join(config['paths']['saved_models'], f'{args.model}_predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Predições salvas em: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['bnn', 'dbnn', 'vanilla'])
    args = parser.parse_args()
    main(args)
