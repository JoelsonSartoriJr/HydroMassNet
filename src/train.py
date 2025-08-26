import argparse
import yaml
import os
import json
import pandas as pd
import tensorflow as tf

from .data import DataHandler
from .models.bayesian_dense_network import BayesianDenseNetwork
from .models.bayesian_density_network import BayesianDensityNetwork
from .models.vanilla_network import VanillaNetwork

def train(model_name, features, exp_config, main_config):
    """Função principal de treinamento unificada e final."""
    data_handler = DataHandler(config=main_config, feature_override=features)
    x_train, y_train, x_val, y_val, _, _, _ = data_handler.get_full_dataset_and_splits()

    input_shape = (len(features),)

    model_map = {
        'bnn': BayesianDenseNetwork,
        'dbnn': BayesianDensityNetwork,
        'vanilla': VanillaNetwork
    }

    if model_name not in model_map:
        raise ValueError(f"Modelo '{model_name}' não reconhecido.")

    model = model_map[model_name](input_shape=input_shape, config=exp_config)

    learning_rate = exp_config['learning_rate']
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',
            patience=main_config['training']['patience'],
            mode='min',
            restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=main_config['training']['epochs'],
        batch_size=exp_config.get('batch_size', 64),
        verbose=0,
        callbacks=callbacks
    )

    results_df = pd.DataFrame(history.history)
    best_epoch_metrics = results_df.loc[results_df['val_mae'].idxmin()]
    print(json.dumps(best_epoch_metrics.to_dict()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Treinamento de Modelos")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--experiment_id', type=str, required=True)
    args = parser.parse_args()

    features_list = [f.strip() for f in args.features.split(',')]
    experiment_config = json.loads(args.config)

    with open('config/config.yaml', 'r') as f:
        main_config_data = yaml.safe_load(f)

    train(
        model_name=args.model_name,
        features=features_list,
        exp_config=experiment_config,
        main_config=main_config_data
    )
