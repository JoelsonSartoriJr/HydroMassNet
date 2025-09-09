import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation
from .bayesian_dense_layer import BayesianDenseLayer
from ..utils.train_utils import parse_layers

class BayesianDenseNetwork(tf.keras.Model):
    """Rede Neural Densa Bayesiana com lógica de treinamento otimizada."""

    def __init__(self, n_features: int, layers_config: str, learning_rate: float, name: str = 'bnn'):
        super(BayesianDenseNetwork, self).__init__(name=name)
        self.n_features = n_features
        self.layer_dims = parse_layers(layers_config, n_features, output_dim=1)

        self.bnn_layers = []
        for i, (d_in, d_out) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            self.bnn_layers.append(BayesianDenseLayer(d_in, d_out, name=f'{name}_dense_{i}'))
            if i < len(self.layer_dims) - 2:
                self.bnn_layers.append(BatchNormalization(name=f'{name}_batch_norm_{i}'))
                self.bnn_layers.append(Activation('relu', name=f'{name}_activation_{i}'))

        self.learning_rate = learning_rate

    def get_model(self):
        """Constrói e compila o modelo Keras."""
        inputs = tf.keras.Input(shape=(self.n_features,))
        outputs = self.call(inputs, sampling=True) # Conecta o call para construção
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        
        # Otimiza o optimizer para precisão mista, se habilitada
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        model.compile(optimizer=optimizer, loss=self.elbo_loss)
        return model

    @tf.function
    def elbo_loss(self, y_true, y_pred):
        """Negative Evidence Lower Bound (ELBO) loss."""
        # Garante que os tipos são consistentes para o cálculo da perda
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        kl_loss = sum(self.losses) / self.n_features # Normaliza KL pelo número de features
        
        # A loss precisa ser float32 para o LossScaleOptimizer
        return tf.cast(mse_loss + kl_loss, tf.float32)

    @tf.function
    def call(self, x, sampling=True):
        """Execução forward do modelo."""
        for layer in self.bnn_layers:
            if isinstance(layer, BayesianDenseLayer):
                x = layer(x, sampling=sampling)
            elif isinstance(layer, BatchNormalization):
                x = layer(x, training=sampling)
            else:
                x = layer(x)
        return x