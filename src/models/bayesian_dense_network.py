import tensorflow as tf
from .bayesian_dense_layer import BayesianDenseLayer
from tensorflow.keras.layers import BatchNormalization, Activation
from ..utils.train_utils import parse_layers

class BayesianDenseNetwork(tf.keras.Model):
    """Rede Neural Densa Bayesiana com lógica de treinamento customizada."""

    def __init__(self, input_shape: tuple, config: dict, name: str = 'bnn', output_dim: int = 1):
        super(BayesianDenseNetwork, self).__init__(name=name)

        input_dim = input_shape[0]
        layers_str = config.get('layers', '128-64')
        self.layer_dims = parse_layers(layers_str, input_dim, output_dim=output_dim)

        self.steps = []
        for i, (d_in, d_out) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            self.steps.append(BayesianDenseLayer(d_in, d_out, name=f'{name}_dense_{i}'))
            if i < len(self.layer_dims) - 2:
                self.steps.append(BatchNormalization(name=f'{name}_batch_norm_{i}'))
                self.steps.append(Activation('relu', name=f'{name}_activation_{i}'))

    def call(self, x, sampling=True):
        for step in self.steps:
            if isinstance(step, BayesianDenseLayer):
                x = step(x, sampling=sampling)
            elif isinstance(step, BatchNormalization):
                x = step(x, training=sampling)
            else:
                x = step(x)
        return x

    def compile(self, optimizer):
        super(BayesianDenseNetwork, self).compile()
        self.optimizer = optimizer
        # --- CORREÇÃO: Usa a classe de perda em vez da função ---
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, sampling=True)
            mse_loss = self.loss_fn(y, y_pred)
            kl_loss = sum(self.losses)
            total_loss = mse_loss + kl_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.mae_metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, sampling=False)
        mse_loss = self.loss_fn(y, y_pred)
        kl_loss = sum(self.losses)
        total_loss = mse_loss + kl_loss

        self.loss_tracker.update_state(total_loss)
        self.mae_metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
