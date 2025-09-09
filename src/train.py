import argparse
import json
import yaml
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import sys
import os
import logging
from datetime import datetime

# Adiciona o diretório raiz ao path e configura logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

from src.data import DataHandler
from src.models.vanilla_network import VanillaNetwork
from src.models.bayesian_dense_network import BayesianDenseNetwork
from src.models.bayesian_density_network import BayesianDensityNetwork

def setup_gpu_strategy(enable_mixed_precision):
    """Configura a estratégia de GPU, incluindo precisão mista se disponível e habilitado."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"{len(gpus)} GPUs encontradas e configuradas com memory growth.")

            if enable_mixed_precision:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info("Política de precisão mista ('mixed_float16') ativada.")

        except RuntimeError as e:
            logger.error(f"Erro ao configurar GPUs: {e}", exc_info=True)
    else:
        logger.warning("Nenhuma GPU encontrada. O treinamento será executado na CPU.")


def get_model(model_type, model_config, n_features):
    """Seleciona e instancia o modelo correto com base no tipo."""
    logger.info(f"Criando modelo do tipo '{model_type}' com n_features={n_features}")
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
    """Ponto de entrada para o treinamento do modelo."""
    parser = argparse.ArgumentParser(description="Treinamento de Modelos")
    parser.add_argument('--model_type', type=str, required=True, help='Tipo de modelo (vanilla, bnn, dbnn)')
    parser.add_argument('--model_config', type=str, required=True, help='Configuração do modelo em formato JSON')
    parser.add_argument('--save_path', type=str, required=True, help='Caminho para salvar o modelo e pesos')
    args = parser.parse_args()

    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        setup_gpu_strategy(config['training'].get('enable_mixed_precision', False))

        model_config = json.loads(args.model_config)
        features = model_config['features']
        batch_size = model_config['batch_size']

        data_handler = DataHandler(config_path='config.yaml')
        train_dataset, val_dataset, _ = data_handler.load_and_prepare_data(features, batch_size)

        model = get_model(args.model_type, model_config, data_handler.n_features)
        model.summary(print_fn=logger.info)

        log_dir = os.path.join("logs", "tensorboard", f"{args.model_type}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        logger.info(f"Logs do TensorBoard serão salvos em: {log_dir}")

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config['training']['patience'],
                verbose=1,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=f"{args.save_path}.weights.h5",
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            ),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

        logger.info(f"Iniciando treinamento do modelo {args.model_type}...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config['training']['epochs'],
            callbacks=callbacks,
            verbose=2
        )

        val_mae_key = next((k for k in history.history if 'val_mean_absolute_error' in k or 'val_mae' in k), 'val_loss')
        final_val_metric = min(history.history[val_mae_key])
        logger.info(f"Treinamento concluído. Métrica de validação final ({val_mae_key}): {final_val_metric:.4f}")

        # Salva o resultado para ser capturado por outros scripts
        performance = {'val_mae': final_val_metric}
        print(json.dumps(performance))

    except Exception as e:
        logger.critical(f"Falha no script de treinamento: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()