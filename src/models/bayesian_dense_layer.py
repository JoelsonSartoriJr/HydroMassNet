import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class BayesianDenseLayer(tf.keras.layers.Layer):
    """Camada Densa Bayesiana implementando a Local Reparameterization Trick.

        Attributes
        ----------
        d_in : int
            Dimensão de entrada da camada.
        d_out : int
            Dimensão de saída da camada.
    """
    def __init__(self, d_in, d_out, name=None):
        super(BayesianDenseLayer, self).__init__(name=name)
        self.d_in = d_in
        self.d_out = d_out

    def build(self, input_shape):
        w_loc_init = tf.keras.initializers.GlorotUniform()
        self.w_loc = self.add_weight(
            name='w_loc', shape=[self.d_in, self.d_out], initializer=w_loc_init
        )
        w_rho_init = tf.keras.initializers.Constant(-5.0)
        self.w_rho = self.add_weight(
            name='w_rho', shape=[self.d_in, self.d_out], initializer=w_rho_init
        )

        b_loc_init = tf.keras.initializers.Zeros()
        self.b_loc = self.add_weight(
            name='b_loc', shape=[self.d_out], initializer=b_loc_init
        )
        b_rho_init = tf.keras.initializers.Constant(-5.0)
        self.b_rho = self.add_weight(
            name='b_rho', shape=[self.d_out], initializer=b_rho_init
        )
        super(BayesianDenseLayer, self).build(input_shape)

    def call(self, x, sampling=True):
        w_scale = tf.nn.softplus(self.w_rho)
        b_scale = tf.nn.softplus(self.b_rho)

        w_prior = tfd.Normal(loc=0., scale=1.)
        b_prior = tfd.Normal(loc=0., scale=1.)

        w_posterior = tfd.Normal(loc=self.w_loc, scale=w_scale)
        b_posterior = tfd.Normal(loc=self.b_loc, scale=b_scale)

        kl_loss = (
            tf.reduce_sum(tfd.kl_divergence(w_posterior, w_prior)) +
            tf.reduce_sum(tfd.kl_divergence(b_posterior, b_prior))
        )

        batch_size = tf.cast(tf.shape(x)[0], dtype=tf.float32)
        self.add_loss(kl_loss / batch_size)

        if sampling:
            activation_loc = tf.matmul(x, self.w_loc) + self.b_loc
            activation_var = tf.matmul(tf.square(x), tf.square(w_scale)) + tf.square(b_scale)
            activation_scale = tf.sqrt(activation_var + 1e-6)

            eps = tf.random.normal(tf.shape(activation_loc), dtype=tf.float32)
            return activation_loc + activation_scale * eps
        else:
            return tf.matmul(x, self.w_loc) + self.b_loc
