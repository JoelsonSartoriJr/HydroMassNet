import tensorflow as tf

class LrHistory(tf.keras.callbacks.Callback):
    def __init__(self, lr_history):
        super().__init__()
        self.lr_history = lr_history

    def on_epoch_end(self, epoch, logs=None):
        self.lr_history.append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))
