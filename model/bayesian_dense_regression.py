import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from model.bayesian_dense_network import BayesianDenseNetwork

tfd = tfp.distributions

class BayesianDenseRegression(tf.keras.Model):
    def __init__(self, layer_dims, name=None):
        super(BayesianDenseRegression, self).__init__(name=name)
        self.layer_dims = layer_dims
        self.loc_net = BayesianDenseNetwork(layer_dims)
        self.std_alpha = tf.Variable([10.0], name='std_alpha')
        self.std_beta = tf.Variable([10.0], name='std_beta')

    def build(self, input_shape):
        self.loc_net.build(input_shape)
        super(BayesianDenseRegression, self).build(input_shape)

    def call(self, x, sampling=True):
        loc_preds = self.loc_net(x, sampling=sampling)
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)
        transform = lambda x: tf.sqrt(tf.math.reciprocal(x))
        N = x.shape[0]
        if sampling:
            std_preds = transform(posterior.sample([N]))
        else:
            std_preds = tf.ones([N, 1]) * transform(posterior.mean())
        return tf.concat([loc_preds, std_preds], 1)

    def log_likelihood(self, x, y, sampling=True):
        preds = self.call(x, sampling=sampling)
        return tfd.Normal(preds[:, 0], preds[:, 1]).log_prob(y[:, 0])

    @tf.function
    def sample(self, x):
        preds = self.call(x)
        return tfd.Normal(preds[:, 0], preds[:, 1]).sample()

    def samples(self, x, n_samples=1):
        samples = np.zeros((x.shape[0], n_samples))
        for i in range(n_samples):
            samples[:, i] = self.sample(x)
        return samples

    @property
    def losses(self):
        net_loss = self.loc_net.losses
        posterior = tfd.Gamma(self.std_alpha, self.std_beta)
        prior = tfd.Gamma(10.0, 10.0)
        std_loss = tfd.kl_divergence(posterior, prior)
        return net_loss + std_loss
