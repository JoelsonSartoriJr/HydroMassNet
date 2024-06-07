import tensorflow as tf
from model.bayesian_dense_layer import BayesianDenseLayer
import tensorflow_probability as tfp

tfd = tfp.distributions

class BayesianDensityNetwork(tf.keras.Model):
    def __init__(self, layer_dims, output_dims, name=None):
        super(BayesianDensityNetwork, self).__init__(name=name)

        self.steps = []
        self.acts = []

        # Inicializa as camadas escondidas e suas funções de ativação
        for i in range(len(layer_dims) - 1):
            self.steps.append(BayesianDenseLayer(layer_dims[i], layer_dims[i + 1]))
            self.acts.append(tf.nn.relu)

        # Inicializa as camadas de saída e suas funções de ativação
        for i in range(len(output_dims) - 1):
            self.steps.append(BayesianDenseLayer(output_dims[i], output_dims[i + 1]))
            self.acts.append(tf.nn.relu)

        # Última camada sem função de ativação
        self.acts[-1] = lambda x: x

    def call(self, x, sampling=True):
        """Realiza a propagação direta na rede."""
        for step, act in zip(self.steps, self.acts):
            x = step(x, sampling=sampling)
            x = act(x)
        return x

    @property
    def kl_loss(self):
        """Calcula a perda de divergência KL para todas as camadas."""
        return tf.reduce_sum([layer.losses for layer in self.steps])

    def log_likelihood(self, x, y):
        """Calcula a log-verossimilhança dos dados."""
        y_pred = self(x, sampling=False)
        return tf.reduce_mean(tfd.Normal(y_pred, 1).log_prob(y))
