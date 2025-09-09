# file: ./src/train.py
import argparse
import json
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import os

# Adiciona o diretório raiz ao path para resolver importações relativas
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataHandler
from src.models.vanilla_network import VanillaNetwork
from src.models.bayesian_dense_network import BayesianDenseNetwork
from src.models.bayesian_density_network import BayesianDensityNetwork

def get_model(model_type, model_config, n_features):
    """
    Seleciona e instancia o modelo correto com base no tipo.
    """
    if model_type == 'vanilla':
        return VanillaNetwork(
            n_features=n_features,
            layers_config=model_config['layers'],
            dropout_rate=model_config.get('dropout', 0.0),
            learning_rate=model_config['learning_rate']
        ).get_model()
    elif model_type == 'bnn':
        return BayesianDenseNetwork(
            n_features=n_features,
            layers_config=model_config['layers'],
            learning_rate=model_config['learning_rate']
        ).get_model()
    elif model_type == 'dbnn':
        return BayesianDensityNetwork(
            n_features=n_features,
            core_layers_config=model_config['core_layers'],
            head_layers_config=model_config['head_layers'],
            learning_rate=model_config['learning_rate']
        ).get_model()
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

def main():
    """
    Ponto de entrada para o treinamento do modelo.
    """
    parser = argparse.ArgumentParser(description="Treinamento de Modelos")
    parser.add_argument('--model_type', type=str, required=True, help='Tipo de modelo (vanilla, bnn, dbnn)')
    parser.add_argument('--model_config', type=str, required=True, help='Configuração do modelo em formato JSON')
    parser.add_argument('--save_path', type=str, required=True, help='Caminho para salvar o modelo e pesos')
    args = parser.parse_args()

    model_config = json.loads(args.model_config)
    features = model_config['features']

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Dispositivo de treinamento: {tf.test.gpu_device_name() if tf.test.gpu_device_name() else '/physical_device:CPU:0'}")

    data_handler = DataHandler(config_path='config.yaml')
    X_train, y_train, X_val, y_val, _, _ = data_handler.load_and_prepare_data(features)

    n_features = X_train.shape[1]
    model = get_model(args.model_type, model_config, n_features)

    print(f"Iniciando treinamento do modelo {args.model_type}...")
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=config['training']['patience'], 
        verbose=1, 
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=f"{args.save_path}.weights.h5",
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training']['epochs'],
        batch_size=model_config['batch_size'],
        callbacks=[early_stopping, model_checkpoint],
        verbose=2
    )
    
    final_val_mae = min(history.history['val_mean_absolute_error'])
    print(f"\nTreinamento concluído. MAE de validação final: {final_val_mae:.4f}")

    # Salva o resultado para ser capturado pelo script de seleção de features
    performance = {'val_mae': final_val_mae}
    print(json.dumps(performance))

if __name__ == "__main__":
    main()
