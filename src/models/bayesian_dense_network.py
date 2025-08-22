import tensorflow as tf
from .bayesian_dense_layer import BayesianDenseLayer

class BayesianDenseNetwork(tf.keras.Model):
    """
    Uma rede neural densa totalmente Bayesiana.

    Esta classe constrói uma rede neural onde cada camada é uma `BayesianDenseLayer`.
    Isso significa que tanto os pesos quanto os biases da rede são tratados como
    distribuições de probabilidade, permitindo a captura da incerteza do modelo.
    """
    def __init__(self, layer_dims, name=None):
        """
        Inicializa a rede Bayesiana.

        Args:
            layer_dims (list of int): Uma lista de inteiros definindo o número de
                                      neurônios em cada camada. Por exemplo,
                                      [13, 64, 32, 1] cria uma rede com uma
                                      camada de entrada de 13 features, duas
                                      camadas ocultas de 64 e 32 neurônios,
                                      e uma camada de saída com 1 neurônio.
            name (str, optional): Nome do modelo. Defaults to None.
        """
        super(BayesianDenseNetwork, self).__init__(name=name)
        self.layer_dims = layer_dims
        self.steps = [] # Lista para armazenar as camadas da rede

    def build(self, input_shape):
        """
        Constrói as camadas da rede com base nas dimensões fornecidas.

        Este método é chamado automaticamente pelo TensorFlow na primeira vez
        que o modelo é executado.
        """
        # Itera sobre as dimensões para criar cada camada densa Bayesiana
        for i in range(len(self.layer_dims) - 1):
            self.steps.append(
                BayesianDenseLayer(self.layer_dims[i], self.layer_dims[i + 1])
            )
        # Chama o build da classe pai para finalizar a construção
        super(BayesianDenseNetwork, self).build(input_shape)

    def call(self, x, sampling=True):
        """
        Executa a passagem para a frente (forward pass) da rede.

        Args:
            x (tf.Tensor): O tensor de entrada.
            sampling (bool, optional): Se True, amostra novos pesos e biases
                                       a cada chamada (usado durante o treino).
                                       Se False, usa a média dos pesos
                                       (usado para inferência/validação).
                                       Defaults to True.

        Returns:
            tf.Tensor: O tensor de saída da rede.
        """
        # Passa a entrada por cada camada sequencialmente
        for step in self.steps:
            x = step(x, sampling=sampling)
        return x

    @property
    def losses(self):
        """
        Calcula a perda de complexidade total do modelo (KL divergence).

        Esta é a soma das perdas de divergência KL de todas as camadas
        Bayesianas na rede. Esta perda penaliza o modelo por se desviar
        muito da distribuição a priori dos pesos.

        Returns:
            tf.Tensor: Um escalar representando a perda KL total.
        """
        # Soma as perdas de cada camada individual
        return tf.reduce_sum([step.losses for step in self.steps])
