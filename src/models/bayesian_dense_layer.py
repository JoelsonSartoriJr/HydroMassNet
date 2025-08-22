import tensorflow as tf
import tensorflow_probability as tfp
from func.utils import xavier

tfd = tfp.distributions

class BayesianDenseLayer(tf.keras.layers.Layer):
    def __init__(self, d_in, d_out, name=None):
        super(BayesianDenseLayer, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out
        self.w_loc = None
        self.w_std = None
        self.b_loc = None
        self.b_std = None

    def build(self, input_shape):
        self.w_loc = self.add_weight(name='w_loc', shape=[self.d_in, self.d_out], initializer=xavier)
        self.w_std = self.add_weight(name='w_std', shape=[self.d_in, self.d_out], initializer=tf.constant_initializer(-6.0))
        self.b_loc = self.add_weight(name='b_loc', shape=[1, self.d_out], initializer=xavier)
        self.b_std = self.add_weight(name='b_std', shape=[1, self.d_out], initializer=tf.constant_initializer(-6.0))
        super(BayesianDenseLayer, self).build(input_shape)

    @property
    def weight(self):
        return tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std))

    @property
    def bias(self):
        return tfd.Normal(self.b_loc, tf.nn.softplus(self.b_std))

    def call(self, x, sampling=True):
        if sampling:
            return x @ self.weight.sample() + self.bias.sample()
        else:
            return x @ self.w_loc + self.b_loc

    @property
    def losses(self):
        prior = tfd.Normal(0, 1)
        return (tf.reduce_sum(tfd.kl_divergence(self.weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(self.bias, prior)))
