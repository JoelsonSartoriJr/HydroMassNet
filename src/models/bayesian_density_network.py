import tensorflow as tf
from .bayesian_dense_layer import BayesianDenseLayer

class BayesianDensityNetwork(tf.keras.Model):
    """
    Uma rede que pode ter um núcleo compartilhado e múltiplas cabeças de predição.
    """
    def __init__(self, core_layers: list, head_layers: list, config: dict):
        """
        Args:
            core_layers (list): Dimensões das camadas do núcleo.
            head_layers (list): Dimensões da(s) camada(s) da cabeça.
            config (dict): Dicionário de configuração do projeto.
        """
        super().__init__()
        self.config = config
        self.core_net = tf.keras.Sequential([
            BayesianDenseLayer(units, activation='relu', config=self.config)
            for units in core_layers
        ])
        self.head_net = tf.keras.Sequential([
            BayesianDenseLayer(units, activation=None, config=self.config)
            for units in head_layers
        ])

    def call(self, inputs):
        """Executa a passagem forward."""
        x = self.core_net(inputs)
        output = self.head_net(x)
        return output
