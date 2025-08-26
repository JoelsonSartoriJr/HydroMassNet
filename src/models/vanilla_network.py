import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

class VanillaNetwork(tf.keras.Model):
    """
    Construtor para um modelo de rede neural densa simples (Vanilla).
    """
    def __init__(self, input_shape: tuple, config: dict, name: str = 'vanilla'):
        super(VanillaNetwork, self).__init__(name=name)

        layer_dims = [int(u) for u in config['layers'].split('-')] + [1]
        dropout_rate = config.get('dropout', 0.2)

        layers = [Input(shape=input_shape, name="input_layer")]
        for units in layer_dims[:-1]:
            layers.extend([Dense(units, activation='relu'), Dropout(dropout_rate)])
        layers.append(Dense(layer_dims[-1], name="output_layer"))

        self.model = Sequential(layers, name='vanilla_network')

    def call(self, x):
        """Executa a passagem para a frente (forward pass)."""
        return self.model(x)

    def compile(self, optimizer):
        """Compila o modelo com a perda e métricas padrão."""
        super(VanillaNetwork, self).compile()
        self.optimizer = optimizer
        # O modelo sequencial já tem um loop de treino, então só precisamos compilar.
        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics=['mae'])

    def train_step(self, data):
        return self.model.train_step(data)

    def test_step(self, data):
        return self.model.test_step(data)
