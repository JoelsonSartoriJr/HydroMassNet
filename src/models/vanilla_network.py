import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

def create_vanilla_model(input_shape, layers_config, dropout_rate, learning_rate):
    """
    Cria um modelo de rede neural densa (vanilla).
    """
    inputs = Input(shape=(input_shape,))
    x = inputs
    
    layer_sizes = [int(size) for size in layers_config.split('-')]
    
    for size in layer_sizes:
        x = Dense(size, activation='relu')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
            
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model
