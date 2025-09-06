# file: ./src/train.py
import argparse
import yaml
import os
import json
import tensorflow as tf
import pandas as pd
from src.data import DataHandler
from src.models.bayesian_dense_regression import BayesianDenseRegression
from src.models.bayesian_density_network import BayesianDensityNetwork
from src.models.vanilla_network import create_vanilla_model

def main():
    """
    Ponto de entrada para treinar um modelo com uma configuração específica.
    """
    parser = argparse.ArgumentParser(description="Script unificado para treinamento de modelos.")
    parser.add_argument('--model_type', type=str, required=True, choices=['bnn', 'dbnn', 'vanilla'])
    parser.add_argument('--model_config', type=str, required=True, help='String JSON com a configuração do modelo.')
    parser.add_argument('--save_path', type=str, required=True, help='Caminho base para salvar os artefatos.')
    args = parser.parse_args()

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_cfg = json.loads(args.model_config)
    train_cfg = config['training']

    data_handler = DataHandler(config, feature_override=model_cfg.get('features'))
    x_train, y_train, x_val, y_val, _, _, features_out = data_handler.get_full_dataset_and_splits()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(model_cfg['batch_size']).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(model_cfg['batch_size']).prefetch(tf.data.AUTOTUNE)
    input_shape = (len(features_out),)

    model = None
    if args.model_type == 'bnn':
        layers = [int(x) for x in model_cfg['layers'].split('-')]
        model = BayesianDenseRegression(layers, config)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg['learning_rate']), metrics=['mae'])
    elif args.model_type == 'dbnn':
        core_layers = [int(x) for x in model_cfg['core_layers'].split('-')]
        head_layers = [int(x) for x in str(model_cfg['head_layers']).split('-')]
        model = BayesianDensityNetwork(core_layers, head_layers, config)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg['learning_rate']), metrics=['mae'])
    elif args.model_type == 'vanilla':
        layers = [int(x) for x in model_cfg['layers'].split('-')]
        model = create_vanilla_model(layers, input_shape, model_cfg['dropout'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg['learning_rate']), loss='mean_squared_error', metrics=['mae'])

    model_path = f"{args.save_path}.weights.h5"
    history_path = f"{args.save_path}_history.csv"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=train_cfg['patience'], restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
    ]

    print(f"--- Treinando modelo '{args.model_type}' ---")
    print(json.dumps(model_cfg, indent=2))

    history = model.fit(train_ds, epochs=train_cfg['epochs'], validation_data=val_ds, callbacks=callbacks, verbose=2)
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    print(f"Treinamento concluído. Modelo salvo em {model_path}")
    print(f"Histórico salvo em {history_path}")

    best_val_mae = min(history.history.get('val_mae', [float('inf')]))
    print(json.dumps({"val_mae": best_val_mae}))

if __name__ == "__main__":
    main()
