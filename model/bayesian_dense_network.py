import tensorflow as tf
from model.bayesian_dense_layer import BayesianDenseLayer

class BayesianDenseNetwork(tf.keras.Model):
    def __init__(self, layer_dims, name=None):
        super(BayesianDenseNetwork, self).__init__(name=name)
        self.steps = []
        for i in range(len(layer_dims) - 1):
            self.steps.append(BayesianDenseLayer(layer_dims[i], layer_dims[i + 1]))

    def call(self, x, sampling=True):
        for step in self.steps:
            x = step(x, sampling=sampling)
        return x

    @property
    def losses(self):
        return tf.reduce_sum([step.losses for step in self.steps])
