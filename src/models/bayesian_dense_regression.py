import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from .bayesian_dense_network import BayesianDenseNetwork

tfd = tfp.distributions

class BayesianDenseRegression(tf.keras.Model):
    """Modelo de Regressão Densa Bayesiana que usa uma rede para a média
        e uma distribuição Gamma para a incerteza.

        Attributes
        ----------
        layer_dims : list
            Lista de inteiros com as dimensões de cada camada.
        loc_net : BayesianDenseNetwork
            Rede neural para prever a média da distribuição de saída.
        std_alpha : tf.Variable
            Variável para o parâmetro alfa da distribuição Gamma.
        std_beta : tf.Variable
            Variável para o parâmetro beta da distribuição Gamma.
        std_prior : tfd.Gamma
            Prior da distribuição de incerteza.
        loss_tracker : tf.keras.metrics.Mean
            Métrica para rastrear a perda.
        mae_metric : tf.keras.metrics.MeanAbsoluteError
            Métrica para rastrear o erro absoluto médio.
    """
    def __init__(self, layer_dims, name=None):
        super(BayesianDenseRegression, self).__init__(name=name)
        self.layer_dims = layer_dims
        self.loc_net = BayesianDenseNetwork(layer_dims)
        self.std_alpha = tf.Variable([10.0], dtype=tf.float32, name='std_alpha')
        self.std_beta = tf.Variable([10.0], dtype=tf.float32, name='std_beta')
        self.std_prior = tfd.Gamma(10.0, 10.0)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

    def call(self, x, sampling=True):
        loc_preds = self.loc_net(x, sampling=sampling)
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)

        std_loss = tfd.kl_divergence(posterior, self.std_prior)
        self.add_loss(tf.reduce_sum(std_loss))

        transform = lambda val: tf.sqrt(tf.math.reciprocal(val))

        N = tf.shape(x)[0]

        if sampling:
            std_preds = tf.ones([N, 1]) * transform(posterior.sample())
        else:
            std_preds = tf.ones([N, 1]) * transform(posterior.mean())

        return tfd.Normal(loc=loc_preds, scale=std_preds)

    def compile(self, optimizer):
        super(BayesianDenseRegression, self).compile()
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

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred_dist.mean())

        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}

    def test_step(self, data):
        x, y = data
        y_pred_dist = self(x, sampling=False)
        nll = -tf.reduce_mean(y_pred_dist.log_prob(y))
        kl_loss = sum(self.losses)
        loss = nll + kl_loss

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred_dist.mean())

        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result()}
