import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from .bayesian_dense_layer import BayesianDenseLayer
from ..utils.train_utils import parse_layers

tfd = tfp.distributions

class BayesianDensityNetwork(tf.keras.Model):
    """Rede Neural de Densidade Bayesiana."""

    def __init__(self, n_features: int, core_layers_config: str, head_layers_config: str, learning_rate: float, name: str = 'dbnn'):
        super(BayesianDensityNetwork, self).__init__(name=name)
        self.n_features = n_features
        self.learning_rate = learning_rate
        
        # --- Camadas do Core Bayesiano ---
        core_dims = parse_layers(core_layers_config, n_features)
        self.core_layers = []
        for i, (d_in, d_out) in enumerate(zip(core_dims[:-1], core_dims[1:])):
            self.core_layers.append(BayesianDenseLayer(d_in, d_out, name=f'{name}_core_dense_{i}'))
            if i < len(core_dims) - 2:
                self.core_layers.append(BatchNormalization(name=f'{name}_core_bn_{i}'))
                self.core_layers.append(Activation('relu', name=f'{name}_core_relu_{i}'))

        # --- Camadas do Head de Densidade (determinístico) ---
        # A saída do core é a entrada do head
        head_input_dim = core_dims[-1]
        # O head prevê os parâmetros da distribuição (média e desvio padrão)
        head_dims = parse_layers(head_layers_config, head_input_dim, output_dim=2)
        self.head_layers = []
        for i, (d_in, d_out) in enumerate(zip(head_dims[:-1], head_dims[1:])):
            self.head_layers.append(Dense(d_out, name=f'{name}_head_dense_{i}'))
            if i < len(head_dims) - 2: # Ativação em camadas ocultas do head
                 self.head_layers.append(Activation('relu', name=f'{name}_head_relu_{i}'))


    def get_model(self):
        """Constrói e compila o modelo Keras."""
        inputs = tf.keras.Input(shape=(self.n_features,))
        distribution = self.call(inputs, sampling=True)
        # O modelo retorna a distribuição, a perda será calculada sobre ela
        model = tf.keras.Model(inputs=inputs, outputs=distribution)
        
        optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
        # A perda é o log-prob da distribuição + KL do core
        model.compile(optimizer=optimizer, loss=self.dbnn_loss)
        return model

    @tf.function
    def dbnn_loss(self, y_true, y_pred_dist):
        """Calcula a perda combinada."""
        # Negativo Log-Probabilidade
        y_true = tf.cast(y_true, tf.float32)
        log_prob = y_pred_dist.log_prob(y_true)
        nll_loss = -tf.reduce_mean(log_prob)

        # Perda KL das camadas do core
        kl_loss = sum(self.losses) / self.n_features

        total_loss = nll_loss + kl_loss
        return tf.cast(total_loss, tf.float32)
        
    @tf.function
    def call(self, x, sampling=True):
        """Execução forward do modelo."""
        # Passa pelo core bayesiano
        core_output = x
        for layer in self.core_layers:
            if isinstance(layer, BayesianDenseLayer):
                core_output = layer(core_output, sampling=sampling)
            elif isinstance(layer, BatchNormalization):
                core_output = layer(core_output, training=sampling)
            else:
                core_output = layer(core_output)

        # Passa pelo head determinístico
        head_output = core_output
        for layer in self.head_layers:
            head_output = layer(head_output)
            
        # A saída do head são os parâmetros da distribuição Normal
        # loc (média), scale (desvio padrão)
        loc, scale = tf.unstack(head_output, num=2, axis=-1)
        
        # Garante que a escala (desvio padrão) seja positiva
        scale = tf.math.softplus(scale) + 1e-6
        
        return tfd.Normal(loc=loc, scale=scale)
