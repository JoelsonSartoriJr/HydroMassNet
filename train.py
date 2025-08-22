import argparse
import yaml
import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import r2_score

from src.data import DataHandler
from src.models.bayesian_dense_regression import BayesianDenseRegression
from src.models.bayesian_density_network import BayesianDensityNetwork
from src.plot import plot_metrics

# Base Trainer Class para encapsular a lógica de treino
class BayesianTrainer(tf.keras.Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

    @tf.function # Compila o passo de treino para alta performance
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Pega a predição (distribuição) do modelo
            y_pred_dist = self.model(x, sampling=True)
            # Calcula a perda de log-probabilidade negativa
            nll = -tf.reduce_mean(y_pred_dist.log_prob(y))
            # Pega a perda de complexidade do modelo (KL)
            kl_loss = self.model.losses
            # A perda final é a soma (ELBO)
            loss = nll + kl_loss

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Atualiza as métricas
        self.loss_tracker.update_state(loss)
        # Para o MAE, usamos a média da distribuição preditiva
        self.mae_metric.update_state(y, y_pred_dist.mean())

        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        y_pred_dist = self.model(x, sampling=False)
        nll = -tf.reduce_mean(y_pred_dist.log_prob(y))
        kl_loss = self.model.losses
        loss = nll + kl_loss

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred_dist.mean())

        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Criação de diretórios
    os.makedirs(config['paths']['saved_models'], exist_ok=True)
    os.makedirs(config['paths']['plots'], exist_ok=True)

    # Carregamento e processamento de dados
    data_handler = DataHandler(config)
    data_scaled_x, data_scaled_y = data_handler.load_and_preprocess()
    x_train, y_train, x_val, y_val = data_handler.split_data(data_scaled_x, data_scaled_y)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(config['training']['batch_size']).prefetch(tf.data.AUTOTUNE)

    input_shape = (x_train.shape[1],)

    # Seleção e construção do modelo
    model_instance = None
    if args.model == 'bnn':
        model_config = config['models']['bnn']
        model_instance = BayesianDenseRegression(model_config['layers'])
    elif args.model == 'dbnn':
        model_config = config['models']['dbnn']
        model_instance = BayesianDensityNetwork(model_config['core_layers'], model_config['head_layers'])
    else:
        raise ValueError(f"Modelo '{args.model}' não é suportado.")

    model_instance.build(input_shape)

    # Encapsula o modelo no Trainer
    trainer = BayesianTrainer(model_instance)

    # Compila o trainer com otimizador
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']))

    # Callbacks para um treinamento mais inteligente
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['paths']['saved_models'], model_config['save_name']),
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(config['paths']['plots'], 'logs'))
    ]

    print(f"--- Treinando o modelo {args.model.upper()} ---")
    history = trainer.fit(
        train_dataset,
        epochs=config['training']['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks
    )

    print(f"Modelo salvo em: {os.path.join(config['paths']['saved_models'], model_config['save_name'])}")

    # Plot final das métricas
    # plot_metrics(history.history['loss'], history.history['mae'], r2_placeholder, args.model.upper(), config['paths']['plots'])
    # Nota: O cálculo de R2 precisa ser feito após o treino no conjunto de validação.
    # A implementação exata do plot pode precisar de ajuste para o formato do 'history.history'.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinador de Modelos para HydroMassNet")
    parser.add_argument('--model', type=str, required=True, choices=['bnn', 'dbnn'], help='Qual modelo treinar.')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Caminho para o arquivo de configuração.')
    args = parser.parse_args()
    main(args)
