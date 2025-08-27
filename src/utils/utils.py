import tensorflow as tf

def print_model_summary(model, input_shape):
    """
    Imprime um resumo do modelo.

    Parameters
    ----------
    model : tf.keras.Model
        O modelo do TensorFlow.
    input_shape : tuple
        O formato de entrada do modelo.
    """
    model.build(input_shape)
    model.summary()

def mean_absolute_error(y_true, y_pred):
    """
    Calcula o Erro Absoluto Médio.

    Parameters
    ----------
    y_true : np.ndarray
        Valores verdadeiros.
    y_pred : np.ndarray
        Valores preditos.

    Returns
    -------
    float
        O valor do MAE.
    """
    return tf.keras.metrics.MeanAbsoluteError()(y_true, y_pred).numpy()
