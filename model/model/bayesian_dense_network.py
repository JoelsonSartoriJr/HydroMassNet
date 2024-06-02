import tensorflow as tf
from model.bayesian_dense_layer import BayesianDenseLayer

class BayesianDenseNetwork(tf.keras.Model):
    def __init__(self, dims, name=None):
        super(BayesianDenseNetwork, self).__init__(name=name)
        self.steps = []
        self.acts = []
        for i in range(len(dims) - 1):
            self.steps += [BayesianDenseLayer(dims[i], dims[i + 1])]
            self.acts += [tf.nn.relu]
        self.acts[-1] = lambda x: x

    def call(self, x, sampling=True):
        for i in range(len(self.steps)):
            x = self.steps[i](x, sampling=sampling)
            x = self.acts[i](x)
        return x

    @property
    def losses(self):
        return tf.reduce_sum([s.losses for s in self.steps])
