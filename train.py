# file: ./train.py
import argparse
import yaml
import os
import pandas as pd
import tensorflow as tf
from src.hydromassnet.data import DataHandler
from src.hydromassnet.models import model_factory

def main(model_name: str):
    """
    Função principal para treinar um modelo especificado.

    Parameters
    ----------
    model_name : str
        O nome do modelo a ser treinado (deve ser uma chave em config.yaml).
    """
    print(f"--- Iniciando Treinamento para o Modelo: {model_name.upper()} ---")

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_cfg = config['models'][model_name]
    train_cfg = config['training']
    paths_cfg = config['paths']

    print("--- Carregando e preparando os dados ---")
    data_handler = DataHandler(config, feature_override=model_cfg['features'])
    x_train, y_train, x_val, y_val, _, _, features = data_handler.get_full_dataset_and_splits()

    num_train_samples = x_train.shape[0]
    input_shape = len(features)

    print(f"Shape dos dados de treino: {x_train.shape}")
    print(f"Shape dos dados de validação: {x_val.shape}")

    print("--- Construindo o modelo ---")
    model = model_factory(model_cfg, input_shape, num_train_samples)
    model.summary()

    model_dir = paths_cfg['models']
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_name}.weights.h5')

    # Remove existing model file to prevent locking issues
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
        except OSError:
            pass

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=train_cfg['patience'],
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, save_weights_only=True, monitor='val_loss',
            mode='min', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', mode='min', factor=0.5, patience=10,
            min_lr=1e-7, verbose=1
        )
    ]

    print(f"--- Iniciando o treinamento por {train_cfg['epochs']} épocas ---")
    try:
        history = model.fit(
            x_train, y_train,
            epochs=train_cfg['epochs'],
            validation_data=(x_val, y_val),
            batch_size=model_cfg.get('batch_size', 64),
            callbacks=callbacks,
            verbose=2
        )
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        # Try to save manually if checkpoint failed
        try:
            model.save_weights(model_path)
            print(f"Pesos salvos manualmente em: {model_path}")
        except Exception as save_error:
            print(f"Erro ao salvar pesos manualmente: {save_error}")
        raise

    history_df = pd.DataFrame(history.history)
    history_path = os.path.join(paths_cfg['results'], f'{model_name}_history.csv')
    history_df.to_csv(history_path, index=False)

    print(f"\n--- Treinamento concluído para o modelo {model_name.upper()} ---")
    print(f"Melhor 'val_loss': {min(history.history.get('val_loss', [0])):.4f}")
    print(f"Modelo salvo em: {model_path}")
    print(f"Histórico de treinamento salvo em: {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script de Treinamento HydroMassNet')
    parser.add_argument('--model', type=str, required=True,
                        help='Nome do modelo a ser treinado (ex: bnn, dbnn, vanilla).')
    args = parser.parse_args()
    main(args.model)
