import tensorflow as tf
import numpy as np

def xavier(shape, dtype=tf.float32):
    stddev = tf.sqrt(2.0 / tf.reduce_sum(tf.cast(shape, tf.float32)))
    return tf.random.truncated_normal(shape, mean=0.0, stddev=stddev, dtype=dtype)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def print_model_summary(model, input_shape):
    model.build(input_shape)
    model.summary()

def calculate_accuracy(y_true, y_pred, threshold=0.1):
    """Calculate accuracy based on the given threshold.

        Parameters:
            y_true (array): True values.
            y_pred (array): Predicted values.
            threshold (float): Threshold for accuracy calculation.

        Returns:
            float: Calculated accuracy.
    """
    # Adicionar uma pequena constante para evitar divisão por zero
    epsilon = 1e-10
    accuracy = np.mean((np.abs(y_pred - y_true) / (y_true + epsilon)) <= threshold) * 100
    return accuracy
