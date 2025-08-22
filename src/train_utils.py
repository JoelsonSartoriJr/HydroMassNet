import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import r2_score
from model.bayesian_dense_regression import BayesianDenseRegression
from model.bayesian_density_network import BayesianDensityNetwork
from func.utils import print_model_summary, mean_absolute_error
import tensorflow as tf

@tf.function
def train_step(model, optimizer, x_data, y_data):
    """
    Executa um passo de treinamento para os modelos BNN e DBNN.
    """
    with tf.GradientTape() as tape:
        log_likelihoods = model.log_likelihood(x_data, y_data)
        kl_loss = model.losses

        num_samples = tf.cast(tf.shape(x_data)[0], tf.float32)
        elbo_loss = (kl_loss / num_samples) - tf.reduce_mean(log_likelihoods)

    gradients = tape.gradient(elbo_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return elbo_loss

    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    elbo = np.zeros(epochs)
    mae = np.zeros(epochs)
    r2 = np.zeros(epochs)

    @tf.function
    def train_step(model, optimizer, x_data, y_data):
        with tf.GradientTape() as tape:
            log_likelihoods = model.log_likelihood(x_data, y_data)
            kl_loss = model.losses
            if x_data.shape[0] == 0:  # Adicionando verificação para evitar divisão por zero
                elbo_loss = tf.constant(np.inf)
            else:
                elbo_loss = kl_loss / x_data.shape[0] - tf.reduce_mean(log_likelihoods)
        gradients = tape.gradient(elbo_loss, model.trainable_variables)
        if gradients and all(g is not None for g in gradients):
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            raise ValueError("Gradientes não foram calculados. Verifique se o modelo possui variáveis treináveis.")
        return elbo_loss

    for epoch in tqdm(range(epochs), desc=f'Training {model_name}'):
        for x_data, y_data in data_train:
            elbo[epoch] = train_step(model, optimizer, x_data, y_data)

        # Evaluate performance on validation data
        y_true = []
        y_pred = []
        for x_data, y_data in data_val:
            pred = model(x_data, sampling=False)[:, 0].numpy()
            y_pred.extend(pred)
            y_true.extend(y_data.numpy())
            mae[epoch] = mean_absolute_error(pred, y_data.numpy())
        r2[epoch] = r2_score(y_true, y_pred)

    plot_metrics(elbo, mae, r2, model_name)
    return model, mae, r2

def train_bayesian_density_network(core_dims, head_dims, x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, weight_decay, val_split, model_name, scaler_y_fit, input_shape):
    model = BayesianDensityNetwork(core_dims, head_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    print_model_summary(model, input_shape)

    data_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)

    elbo = np.zeros(epochs)
    mae = np.zeros(epochs)
    r2 = np.zeros(epochs)

    @tf.function
    def train_step(model, optimizer, x_data, y_data):
        with tf.GradientTape() as tape:
            log_likelihoods = model.log_likelihood(x_data, y_data)
            kl_loss = model.losses
            if x_data.shape[0] == 0:  # Adicionando verificação para evitar divisão por zero
                elbo_loss = tf.constant(np.inf)
            else:
                elbo_loss = kl_loss / x_data.shape[0] - tf.reduce_mean(log_likelihoods)
        gradients = tape.gradient(elbo_loss, model.trainable_variables)
        if gradients and all(g is not None for g in gradients):
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        else:
            raise ValueError("Gradientes não foram calculados. Verifique se o modelo possui variáveis treináveis.")
        return elbo_loss

    for epoch in tqdm(range(epochs), desc=f'Training {model_name}'):
        for x_data, y_data in data_train:
            elbo[epoch] = train_step(model, optimizer, x_data, y_data)

        # Evaluate performance on validation data
        y_true = []
        y_pred = []
        for x_data, y_data in data_val:
            pred = model(x_data, sampling=False)[:, 0].numpy()
            y_pred.extend(pred)
            y_true.extend(y_data.numpy())
            mae[epoch] = mean_absolute_error(pred, y_data.numpy())
        r2[epoch] = r2_score(y_true, y_pred)

    plot_metrics(elbo, mae, r2, model_name)
    return model, mae, r2
