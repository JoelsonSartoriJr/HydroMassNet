import tensorflow as tf
import tensorflow_probability as tfp
from model.bayesian_dense_network import BayesianDenseNetwork

tfd = tfp.distributions

class BayesianDensityNetwork(tf.keras.Model):
    def __init__(self, core_dims, head_dims, name=None):
        super(BayesianDensityNetwork, self).__init__(name=name)
        self.core_net = BayesianDenseNetwork(core_dims)
        self.loc_net = BayesianDenseNetwork([core_dims[-1]] + head_dims)
        self.std_net = BayesianDenseNetwork([core_dims[-1]] + head_dims)

    def call(self, x, sampling=True):
        x = self.core_net(x, sampling=sampling)
        x = tf.nn.relu(x)
        loc_preds = self.loc_net(x, sampling=sampling)
        std_preds = self.std_net(x, sampling=sampling)
        std_preds = tf.nn.softplus(std_preds)
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
        return self.core_net.losses + self.loc_net.losses + self.std_net.losses
