import argparse
import yaml
import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np

from data import DataHandler
from models.bayesian_dense_network import BayesianDenseNetwork
from models.bayesian_density_network import BayesianDensityNetwork
from models.vanilla_network import VanillaNetwork

def train_final_model(model_name, features, exp_config, main_config):
    """
    Treina o modelo final no conjunto de dados de treino + validação.
    """
    print(f"\n--- Treinando modelo campeão: {model_name.upper()} ---")

    # 1. Carrega os dados
    data_handler = DataHandler(config=main_config, feature_override=features)
    x_train, y_train, x_val, y_val, _, _, _ = data_handler.get_full_dataset_and_splits()

    # 2. Combina os dados de treino e validação
    X_train_full = np.concatenate([x_train, x_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    print(f"Treinando com {X_train_full.shape[0]} amostras.")

    input_shape = (len(features),)

    # 3. Mapeia e constrói o modelo
    model_map = {
        'bnn': BayesianDenseNetwork,
        'dbnn': BayesianDensityNetwork,
        'vanilla': VanillaNetwork
    }
    model = model_map[model_name](input_shape=input_shape, config=exp_config)

    learning_rate = exp_config['learning_rate']
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer)

    # 4. Treina o modelo no dataset completo (sem callbacks de validação)
    model.fit(
        X_train_full, y_train_full,
        epochs=main_config['training']['epochs'],
        batch_size=exp_config.get('batch_size', 64),
        verbose=1
    )

    # 5. Salva o modelo final
    model_dir = main_config['paths']['saved_models']
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"champion_{model_name}.weights.h5")
    model.save_weights(save_path)

    print(f"--- Modelo campeão salvo em: {save_path} ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Treinamento Final")
    parser.add_argument('--config_path', type=str, required=True, help='Caminho para o arquivo champion_config.yaml')
    args = parser.parse_args()

    # Carrega a configuração do campeão
    with open(args.config_path, 'r') as f:
        champion_config = yaml.safe_load(f)['champion_model']

    # Carrega a configuração principal
    with open('config/config.yaml', 'r') as f:
        main_config_data = yaml.safe_load(f)

    train_final_model(
        model_name=champion_config['model'],
        features=champion_config['features'],
        exp_config=champion_config,
        main_config=main_config_data
    )
