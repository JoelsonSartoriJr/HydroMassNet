import tensorflow as tf
import tensorflow_probability as tfp
from .bayesian_dense_network import BayesianDenseNetwork
from ..utils.train_utils import parse_layers

tfd = tfp.distributions

class BayesianDensityNetwork(tf.keras.Model):
    """Rede de Densidade Bayesiana."""

    def __init__(self, input_shape: tuple, config: dict, name: str = 'dbnn'):
        """Inicializa a rede, aceitando input_shape e config."""
        super(BayesianDensityNetwork, self).__init__(name=name)

        input_dim = input_shape[0]

        core_layers_str = config.get('core_layers', '256-128')
        core_layer_dims = parse_layers(core_layers_str, input_dim, output_dim=None)

        head_input_dim = core_layer_dims[-1]
        head_layers_str = config.get('head_layers', '64')

        core_config = {'layers': core_layers_str}
        head_config = {'layers': head_layers_str}

        self.core_net = BayesianDenseNetwork(input_shape=input_shape, config=core_config, name=f'{name}_core', output_dim=None)
        self.loc_net = BayesianDenseNetwork(input_shape=(head_input_dim,), config=head_config, name=f'{name}_loc_head')
        self.std_net = BayesianDenseNetwork(input_shape=(head_input_dim,), config=head_config, name=f'{name}_std_head')

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def call(self, x, sampling=True):
        x = self.core_net(x, sampling=sampling)
        loc_preds = self.loc_net(x, sampling=sampling)
        std_preds = tf.nn.softplus(self.std_net(x, sampling=sampling)) + 1e-6
        return tfd.Normal(loc=loc_preds, scale=std_preds)

    def compile(self, optimizer):
        super(BayesianDensityNetwork, self).compile()
        self.optimizer = optimizer

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred_dist = self(x, sampling=True)
            nll = -tf.reduce_mean(y_pred_dist.log_prob(y))
            kl_loss = sum(self.losses)
            loss = nll + kl_loss

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred_dist.mean())
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred_dist = self(x, sampling=False)
        nll = -tf.reduce_mean(y_pred_dist.log_prob(y))
        kl_loss = sum(self.losses)
        loss = nll + kl_loss

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred_dist.mean())
        return {m.name: m.result() for m in self.metrics}
