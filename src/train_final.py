import argparse
import yaml
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import pandas as pd
from src.data import DataHandler
from src.models.bayesian_dense_regression import BayesianDenseRegression
from src.models.bayesian_density_network import BayesianDensityNetwork

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
    Função principal para treinar os modelos campeões.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['champion_models'][args.model]
    train_cfg = config['training']

    data_handler = DataHandler(config, feature_override=model_cfg['features'])
    x_train, y_train, x_val, y_val, _, _, features = data_handler.get_full_dataset_and_splits()

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(model_cfg['batch_size']).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(model_cfg['batch_size']).prefetch(tf.data.AUTOTUNE)

    input_shape = (len(features),)

    if args.model == 'bnn':
        model = BayesianDenseRegression(model_cfg['layers'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg['learning_rate']))
    elif args.model == 'dbnn':
        model = BayesianDensityNetwork(model_cfg['core_layers'], model_cfg['head_layers'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg['learning_rate']))
    elif args.model == 'vanilla':
        layers_to_create = model_cfg['layers'][1:]
        model = create_vanilla_model(layers_to_create, input_shape, model_cfg['dropout'])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_cfg['learning_rate']),
                      loss='mean_squared_error', metrics=['mae'])

    model_path = os.path.join(config['paths']['saved_models'], model_cfg['save_name'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=train_cfg['patience'], restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
    ]

    print(f"--- Treinando o modelo campeão '{args.model}' ---")
    history = model.fit(train_ds, epochs=train_cfg['epochs'], validation_data=val_ds, callbacks=callbacks)

    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(config['paths']['saved_models'], f'{args.model}_history.csv')
    history_df.to_csv(history_path, index=False)

    print(f"Treinamento concluído. Modelo salvo em {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['bnn', 'dbnn', 'vanilla'])
    args = parser.parse_args()
    main(args)
