import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class BayesianDenseLayer(tf.keras.layers.Layer):
    def __init__(self, d_in, d_out, name=None):
        super(BayesianDenseLayer, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out

        # Initialize the means and standard deviations for weights and biases
        self.w_loc = self.add_weight(name='w_loc', shape=[d_in, d_out], initializer=tf.keras.initializers.GlorotNormal())
        self.w_std = self.add_weight(name='w_std', shape=[d_in, d_out], initializer=tf.keras.initializers.GlorotNormal())
        self.b_loc = self.add_weight(name='b_loc', shape=[1, d_out], initializer=tf.keras.initializers.GlorotNormal())
        self.b_std = self.add_weight(name='b_std', shape=[1, d_out], initializer=tf.keras.initializers.GlorotNormal())

        # Adjust the initial value of w_std and b_std
        self.w_std.assign(self.w_std - 6.0)
        self.b_std.assign(self.b_std - 6.0)

    @property
    def weight(self):
        """Returns a normal distribution for the weights."""
        return tfd.Normal(self.w_loc, tf.nn.softplus(self.w_std))

    @property
    def bias(self):
        """Returns a normal distribution for the biases."""
        return tfd.Normal(self.b_loc, tf.nn.softplus(self.b_std))

    def call(self, x, sampling=True):
        """Computes the forward pass."""
        if sampling:
            return x @ self.weight.sample() + self.bias.sample()
        else:
            return x @ self.w_loc + self.b_loc

    @property
    def losses(self):
        """Computes the KL divergence loss between the weight/bias distributions and the prior."""
        prior = tfd.Normal(0, 1)
        return (tf.reduce_sum(tfd.kl_divergence(self.weight, prior)) +
                tf.reduce_sum(tfd.kl_divergence(self.bias, prior)))
