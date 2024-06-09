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
