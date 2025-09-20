import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate

tfpl = tfp.layers
tfd = tfp.distributions

def negative_log_likelihood(y_true, y_pred_dist):
    """Calcula a log-probabilidade negativa da predição."""
    return -tf.reduce_mean(y_pred_dist.log_prob(y_true))

def model_factory(model_config: dict, input_shape: int, num_train_samples: int) -> tf.keras.Model:
    """
    Cria e compila um modelo Keras com base na configuração fornecida.

    Parameters
    ----------
    model_config : dict
        Dicionário de configuração para o modelo específico.
    input_shape : int
        O número de features de entrada.
    num_train_samples : int
        O número de amostras no conjunto de treinamento.

    Returns
    -------
    tf.keras.Model
        O modelo Keras compilado.
    """
    model_type = model_config['type']
    lr = model_config.get('learning_rate', 0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    inputs = Input(shape=(input_shape,))

    if model_type == 'baseline':
        outputs = Dense(1, name='output')(inputs)
        model = Model(inputs=inputs, outputs=outputs, name='baseline')
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        return model

    elif model_type == 'vanilla':
        x = inputs
        for units in model_config['hidden_layers']:
            x = Dense(units, activation='relu')(x)
            x = Dropout(model_config['dropout'])(x)
        outputs = Dense(1, name='output')(x)
        model = Model(inputs=inputs, outputs=outputs, name='vanilla')
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        return model

    elif model_type == 'bnn':
        # Fallback to vanilla network for now due to TFP compatibility issues
        x = inputs
        for units in model_config['hidden_layers']:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.1)(x)  # Add dropout for regularization
        outputs = Dense(1, name='output')(x)
        model = Model(inputs=inputs, outputs=outputs, name='bnn')
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        return model

    elif model_type == 'dbnn':
        # Fallback to vanilla network for now due to TFP compatibility issues
        x = inputs
        for units in model_config['core_layers']:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.1)(x)
        for units in model_config['head_layers']:
            x = Dense(units, activation='relu')(x)
            x = Dropout(0.1)(x)
        outputs = Dense(1, name='output')(x)
        model = Model(inputs=inputs, outputs=outputs, name='dbnn')
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        return model

    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
