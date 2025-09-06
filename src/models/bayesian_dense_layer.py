# file: ./src/models/bayesian_dense_layer.py
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

class BayesianDenseLayer(tf.keras.layers.Layer):
    """
    Camada Densa Bayesiana modernizada, compatível com a API Keras.
    Implementa a "Local Reparameterization Trick".
    """
    def __init__(self, units, activation=None, config=None, **kwargs):
        """
        Inicializador da camada.

        Args:
            units (int): Dimensão de saída da camada (número de neurônios).
            activation (str, optional): Função de ativação a ser usada.
            config (dict, optional): Dicionário de configuração (não utilizado aqui,
                                     mas mantido para compatibilidade de assinatura).
        """
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        # O config não é usado diretamente aqui, mas evita erros de argumento.
        self.config = config

    def build(self, input_shape):
        """
        Cria os pesos da camada com base no formato da entrada.
        Este método é chamado automaticamente pelo Keras.
        """
        # A dimensão de entrada é inferida a partir do último eixo do input_shape
        d_in = input_shape[-1]

        # Pesos (weights)
        self.w_loc = self.add_weight(
            name='w_loc',
            shape=(d_in, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.w_rho = self.add_weight(
            name='w_rho',
            shape=(d_in, self.units),
            initializer=tf.keras.initializers.Constant(-5.0),
            trainable=True,
        )

        # Viés (bias)
        self.b_loc = self.add_weight(
            name='b_loc',
            shape=(self.units,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )
        self.b_rho = self.add_weight(
            name='b_rho',
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(-5.0),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Define a lógica de passagem 'forward' da camada.
        """
        w_scale = tf.nn.softplus(self.w_rho)
        b_scale = tf.nn.softplus(self.b_rho)

        # Adiciona a perda KL (Kullback-Leibler) à coleção de perdas da camada
        # A divergência KL mede a diferença entre a distribuição posterior e a prior.
        w_prior = tfd.Normal(loc=0., scale=1.)
        b_prior = tfd.Normal(loc=0., scale=1.)
        w_posterior = tfd.Normal(loc=self.w_loc, scale=w_scale)
        b_posterior = tfd.Normal(loc=self.b_loc, scale=b_scale)

        kl_loss = (
            tf.reduce_sum(tfd.kl_divergence(w_posterior, w_prior)) +
            tf.reduce_sum(tfd.kl_divergence(b_posterior, b_prior))
        )
        # Normaliza a perda pelo tamanho do batch
        batch_size = tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
        self.add_loss(kl_loss / batch_size)

        # A amostragem só ocorre durante o treinamento
        if training:
            # Local Reparameterization Trick
            activation_loc = tf.matmul(inputs, self.w_loc) + self.b_loc
            activation_var = tf.matmul(tf.square(inputs), tf.square(w_scale)) + tf.square(b_scale)
            activation_scale = tf.sqrt(activation_var + 1e-6)

            # Amostra da distribuição normal para a reparametrização
            eps = tf.random.normal(tf.shape(activation_loc), dtype=tf.float32)
            output = activation_loc + activation_scale * eps
        else:
            # Em modo de inferência/teste, usa-se apenas a média dos pesos
            output = tf.matmul(inputs, self.w_loc) + self.b_loc

        # Aplica a função de ativação, se houver
        if self.activation is not None:
            output = self.activation(output)

        return output
