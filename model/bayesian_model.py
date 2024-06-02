import os
import sys

# Defina a raiz do projeto
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from func.data_preprocessing import load_and_preprocess_data, split_data
from model.bayesian_dense_regression import BayesianDenseRegression
from model.bayesian_density_network import BayesianDensityNetwork

class LrHistory(tf.keras.callbacks.Callback):
    def __init__(self, lr_history):
        super().__init__()
        self.lr_history = lr_history

    def on_epoch_end(self, epoch, logs=None):
        self.lr_history.append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))

def print_model_summary(model, input_shape):
    model.build(input_shape=input_shape)
    model.summary()
    for layer in model.layers:
        print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

def train_bayesian_regression(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, model_name):
    shape = x_train.shape[1]
    model = BayesianDenseRegression([shape, 256, 128, 64, 32, 1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    lr_history = []

    print_model_summary(model, input_shape=(None, x_train.shape[1]))  # Print model summary to check layers

    @tf.function
    def train_step(x_data, y_data):
        with tf.GradientTape() as tape:
            log_likelihoods = model.log_likelihood(x_data, y_data)
            kl_loss = model.losses
            elbo_loss = kl_loss / x_train.shape[0] - tf.reduce_mean(log_likelihoods)
        gradients = tape.gradient(elbo_loss, model.trainable_variables)
        if not gradients:
            print("Gradients are None. Check if the model has trainable variables.")
        if not model.trainable_variables:
            print("Model has no trainable variables.")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return elbo_loss

    elbo = np.zeros(epochs)
    mae = np.zeros(epochs)
    for epoch in range(epochs):
        for x_data, y_data in tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size):
            elbo[epoch] = train_step(x_data, y_data)

        for x_data, y_data in tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0]):
            y_pred = model(x_data, sampling=False)[:, 0]
            mae[epoch] = mean_absolute_error(y_pred, y_data)

    plt.plot(elbo)
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.savefig(f'{model_name}_Elbo.png')
    plt.show()

    plt.plot(mae)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.savefig(f'{model_name}_Mae.png')
    plt.show()

    return model

def train_bayesian_density_network(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, model_name):
    shape = x_train.shape[1]
    model = BayesianDensityNetwork([shape, 256, 128], [64, 32, 1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    lr_history = []

    print_model_summary(model, input_shape=(None, x_train.shape[1]))  # Print model summary to check layers

    @tf.function
    def train_step(x_data, y_data):
        with tf.GradientTape() as tape:
            log_likelihoods = model.log_likelihood(x_data, y_data)
            kl_loss = model.losses
            elbo_loss = kl_loss / x_train.shape[0] - tf.reduce_mean(log_likelihoods)
        gradients = tape.gradient(elbo_loss, model.trainable_variables)
        if not gradients:
            print("Gradients are None. Check if the model has trainable variables.")
        if not model.trainable_variables:
            print("Model has no trainable variables.")
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return elbo_loss

    elbo = np.zeros(epochs)
    mae = np.zeros(epochs)
    for epoch in range(epochs):
        for x_data, y_data in tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size):
            elbo[epoch] = train_step(x_data, y_data)

        for x_data, y_data in tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0]):
            y_pred = model(x_data, sampling=False)[:, 0]
            mae[epoch] = mean_absolute_error(y_pred, y_data)

    plt.plot(elbo)
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.savefig(f'{model_name}_Elbo.png')
    plt.show()

    plt.plot(mae)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.savefig(f'{model_name}_Mae.png')
    plt.show()

    return model

if __name__ == "__main__":
    data_path = os.path.join(project_root, 'data', 'cleaning_data_test.csv')
    val_split = 0.2
    seed = 1601
    batch_size = 2048
    epochs = 100
    learning_rate = 1e-4

    data_scaled_x, data_scaled_y, scaler_x_fit, scaler_y_fit = load_and_preprocess_data(data_path)
    x_train, y_train, x_val, y_val = split_data(data_scaled_x, data_scaled_y, val_split, seed)

    bnn_model = train_bayesian_regression(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, "BNN")
    dbnn_model = train_bayesian_density_network(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, "DBNN")
