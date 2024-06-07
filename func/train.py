from model.bayesian_dense_regression import BayesianDenseRegression
from model.bayesian_density_network import BayesianDensityNetwork
from func.utils import print_model_summary
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_bayesian_regression(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, model_name, scaler_y_fit):
    shape = x_train.shape[1]
    model = BayesianDenseRegression([shape, 256, 128, 64, 32, 1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    lr_history = []

    print_model_summary(model, input_shape=(None, x_train.shape[1]))

    @tf.function
    def train_step(x_data, y_data):
        with tf.GradientTape() as tape:
            log_likelihoods = model.log_likelihood(x_data, y_data)
            kl_loss = model.losses
            elbo_loss = kl_loss / x_train.shape[0] - tf.reduce_mean(log_likelihoods)
        gradients = tape.gradient(elbo_loss, model.trainable_variables)

        if not gradients:
            raise ValueError("Gradientes não foram calculados. Verifique se o modelo possui variáveis treináveis.")
        if not model.trainable_variables:
            raise ValueError("O modelo não possui variáveis treináveis.")

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return elbo_loss

    elbo = np.zeros(epochs)
    mae = np.zeros(epochs)
    accuracy = np.zeros(epochs)

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0])

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_elbo = []

        # Barra de progresso para o epoch
        for x_data, y_data in tqdm(dataset_train, desc=f"Training Epoch {epoch + 1}/{epochs}", unit="batch", leave=False):
            elbo_loss = train_step(x_data, y_data)
            epoch_elbo.append(elbo_loss.numpy())

        elbo[epoch] = np.mean(epoch_elbo)

        for x_data, y_data in dataset_val:
            y_pred = model(x_data, training=False)[:, 0]
            mae[epoch] = mean_absolute_error(y_pred, y_data)
            y_val_real = scaler_y_fit.inverse_transform(y_val).flatten()
            y_pred_real = scaler_y_fit.inverse_transform(y_pred.numpy().reshape(-1, 1)).flatten()
            accuracy[epoch] = np.mean(np.abs(y_pred_real[:len(y_val_real)] - y_val_real) <= 0.1 * y_val_real) * 100

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

    plt.plot(accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.savefig(f'{model_name}_Accuracy.png')
    plt.show()

    model.save(f'{model_name}_model.keras')

    return model

def train_bayesian_density_network(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, model_name, scaler_y_fit):
    shape = x_train.shape[1]
    model = BayesianDensityNetwork([shape, 256, 128], [128, 64, 32, 1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    lr_history = []

    print_model_summary(model, input_shape=(None, x_train.shape[1]))

    @tf.function
    def train_step(x_data, y_data):
        with tf.GradientTape() as tape:
            log_likelihoods = model.log_likelihood(x_data, y_data)
            kl_loss = model.kl_loss
            elbo_loss = kl_loss / x_train.shape[0] - tf.reduce_mean(log_likelihoods)
        gradients = tape.gradient(elbo_loss, model.trainable_variables)

        if not gradients:
            raise ValueError("Gradientes não foram calculados. Verifique se o modelo possui variáveis treináveis.")
        if not model.trainable_variables:
            raise ValueError("O modelo não possui variáveis treináveis.")

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return elbo_loss

    elbo = np.zeros(epochs)
    mae = np.zeros(epochs)
    accuracy = np.zeros(epochs)

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size)
    dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0])

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_elbo = []

        # Barra de progresso para o epoch
        for x_data, y_data in tqdm(dataset_train, desc=f"Training Epoch {epoch + 1}/{epochs}", unit="batch", leave=False):
            elbo_loss = train_step(x_data, y_data)
            epoch_elbo.append(elbo_loss.numpy())

        elbo[epoch] = np.mean(epoch_elbo)

        for x_data, y_data in dataset_val:
            y_pred = model(x_data, training=False)[:, 0]
            mae[epoch] = mean_absolute_error(y_pred, y_data)
            y_val_real = scaler_y_fit.inverse_transform(y_val).flatten()
            y_pred_real = scaler_y_fit.inverse_transform(y_pred.numpy().reshape(-1, 1)).flatten()
            accuracy[epoch] = np.mean(np.abs(y_pred_real[:len(y_val_real)] - y_val_real) <= 0.1 * y_val_real) * 100

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

    plt.plot(accuracy)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.savefig(f'{model_name}_Accuracy.png')
    plt.show()

    model.save(f'{model_name}_model.keras')

    return model
