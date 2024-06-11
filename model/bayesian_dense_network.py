import tensorflow as tf
from model.bayesian_dense_layer import BayesianDenseLayer

class BayesianDenseNetwork(tf.keras.Model):
    def __init__(self, layer_dims, name=None):
        super(BayesianDenseNetwork, self).__init__(name=name)
        self.layer_dims = layer_dims
        self.steps = []

    def build(self, input_shape):
        for i in range(len(self.layer_dims) - 1):
            self.steps.append(BayesianDenseLayer(self.layer_dims[i], self.layer_dims[i + 1]))
        super(BayesianDenseNetwork, self).build(input_shape)

    def call(self, x, sampling=True):
        for step in self.steps:
            x = step(x, sampling=sampling)
        return x

    @property
    def losses(self):
        return tf.reduce_sum([step.losses for step in self.steps])
