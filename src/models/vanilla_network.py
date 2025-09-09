import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from ..utils.train_utils import parse_layers

class VanillaNetwork:
    """
    Classe para construir uma Rede Neural Densa padrão (Vanilla).
    """
    def __init__(self, n_features: int, layers_config: str, dropout_rate: float, learning_rate: float):
        self.n_features = n_features
        self.layers_config = layers_config
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

    def get_model(self):
        """
        Constrói e compila o modelo Keras sequencial.
        """
        # Define as dimensões de entrada, ocultas e de saída
        layer_dims = parse_layers(layers_config_str=self.layers_config, input_dim=self.n_features, output_dim=1)
        
        inputs = Input(shape=(self.n_features,))
        x = inputs
        
        # Adiciona as camadas ocultas
        for i, units in enumerate(layer_dims[1:-1]):
            x = Dense(units, activation='relu', name=f'dense_{i}')(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate, name=f'dropout_{i}')(x)
        
        # Camada de saída
        outputs = Dense(layer_dims[-1], name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
            
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
        
        return model
