import os
import sys
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

def train_bayesian_regression(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, model_name, scaler_y_fit):
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
    accuracy = np.zeros(epochs)
    for epoch in range(epochs):
        for x_data, y_data in tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size):
            elbo[epoch] = train_step(x_data, y_data)

        for x_data, y_data in tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0]):
            y_pred = model(x_data, sampling=False)[:, 0]
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

    return model

def train_bayesian_density_network(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, model_name, scaler_y_fit):
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
    accuracy = np.zeros(epochs)
    for epoch in range(epochs):
        for x_data, y_data in tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(batch_size):
            elbo[epoch] = train_step(x_data, y_data)

        for x_data, y_data in tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0]):
            y_pred = model(x_data, sampling=False)[:, 0]
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

    return model

def make_predictions_and_plot_residuals(model1, model2, data_val, scaler_y_fit):
    # Make predictions on validation data
    for x_data, y_data in data_val:
        resids1 = y_data[:, 0] - model1(x_data, sampling=False)[:, 0]
        resids2 = y_data[:, 0] - model2(x_data, sampling=False)[:, 0]

    # Plot residual distributions
    bins = np.linspace(-2, 2, 100)
    plt.hist(resids1.numpy(), bins, alpha=0.5, label='No Error Estimation')
    plt.hist(resids2.numpy(), bins, alpha=0.5, label='Density Network')
    plt.legend()
    plt.xlabel('Residuals', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Residuo.png')

def plot_predictive_distributions(model1, model2, data_val, scaler_y_fit):
    # Sample from predictive distributions
    for x_data, y_data in data_val:
        samples1 = model1.samples(x_data, 1000)
        samples2 = model2.samples(x_data, 1000)

    y_data_real = scaler_y_fit.inverse_transform(y_data)

    # Plot predictive distributions
    plt.figure(figsize=(20, 10))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        num = np.random.randint(len(y_data_real), size=1)
        samples1_real = scaler_y_fit.inverse_transform(samples1[num, :].reshape(1, -1))[0]
        samples2_real = scaler_y_fit.inverse_transform(samples2[num, :].reshape(1, -1))[0]
        sns.kdeplot(samples1_real, shade=True, label='BNN')
        sns.kdeplot(samples2_real, shade=True, label='DBNN')
        plt.axvline(y_data_real[num], ls=':', color='red', label='True value')
        plt.title(str(i))
        plt.gca().get_yaxis().set_ticklabels([])
        if i < 8:
            plt.gca().get_xaxis().set_ticklabels([])
        if i == 3:
            plt.legend(prop={"size": 12})
    plt.savefig('pred_bnn.png')
    plt.show()

def covered(samples, y_true, prc=95.0):
    """Whether each sample was covered by its predictive interval"""
    q0 = (100.0 - prc) / 2.0  # lower percentile
    q1 = 100.0 - q0  # upper percentile
    within_conf_int = np.zeros(len(y_true))
    for i in range(len(y_true)):
        p0 = np.percentile(samples[i, :], q0)
        p1 = np.percentile(samples[i, :], q1)
        if p0 <= y_true[i] and p1 > y_true[i]:
            within_conf_int[i] = 1
    return within_conf_int

def compute_coverage_and_errors(model1, model2, x_val, y_val, scaler_x_fit, scaler_y_fit):
    # Sample from predictive distributions
    samples1 = model1.samples(x_val, 1000)
    samples2 = model2.samples(x_val, 1000)

    # Compute what samples are covered by their 95% predictive intervals
    covered1 = covered(samples1, y_val)
    covered2 = covered(samples2, y_val)

    mean1, mean2 = [], []

    for i in range(len(samples1)):
        mean1.append(np.mean(samples1[i]))
        mean2.append(np.mean(samples2[i]))

    print(f'Erro médio absoluto BNN {mean_absolute_error(y_val, mean1):.3f}')
    print(f'Score R2 BNN: {r2_score(y_val, mean1) * 100:.3f}')

    print(f'Erro médio absoluto DBNN {mean_absolute_error(y_val, mean2):.3f}')
    print(f'Score R2 DBNN: {r2_score(y_val, mean2) * 100:.3f}')

    mean1 = np.array(mean1).reshape(-1, 1)
    mean2 = np.array(mean2).reshape(-1, 1)
    x_val_real = scaler_x_fit.inverse_transform(x_val)

    star_formation = x_val_real[:, -1]

    preds_real = scaler_y_fit.inverse_transform(mean1).flatten()
    y_val_real = scaler_y_fit.inverse_transform(y_val).flatten()

    plt.figure(figsize=(32, 7))

    error = np.abs(preds_real - y_val_real)  # Ensure no negative values
    plt.plot(y_val_real, y_val_real, color='green', zorder=3)
    plt.errorbar(y_val_real, preds_real, yerr=error, fmt="o", alpha=0.7,)
    plt.scatter(y_val_real, preds_real, c=star_formation, zorder=3, alpha=0.7)
    # labels
    plt.xlabel(r'True values ($log(M_{\odot})$)', fontsize=24)
    plt.ylabel(r'Predict values ($log(M_{\odot})$)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # limits
    plt.xlim(8.5, 10.7)
    plt.ylim(7, 12)
    plt.savefig('realVSpreditoBNN.png')
    plt.show()

    preds_real = scaler_y_fit.inverse_transform(mean2).flatten()
    y_val_real = scaler_y_fit.inverse_transform(y_val).flatten()

    plt.figure(figsize=(32, 7))

    error = np.abs(preds_real - y_val_real)  # Ensure no negative values
    plt.plot(y_val_real, y_val_real, color='green', zorder=3)
    plt.errorbar(y_val_real, preds_real, yerr=error, fmt="o", alpha=0.7,)
    plt.scatter(y_val_real, preds_real, c=star_formation, zorder=3, alpha=0.7)
    # labels
    plt.xlabel(r'True values ($log(M_{\odot})$)', fontsize=24)
    plt.ylabel(r'Predict values ($log(M_{\odot})$)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # limits
    plt.xlim(8.5, 10.7)
    plt.ylim(7, 12)
    plt.savefig('realVSpreditoDBNN.png')
    plt.show()

def plot_accuracy(model1, model2, x_val, y_val, scaler_y_fit):
    preds1 = model1(x_val, sampling=False).numpy().flatten()
    preds2 = model2(x_val, sampling=False).numpy().flatten()

    y_val_real = scaler_y_fit.inverse_transform(y_val).flatten()
    preds1_real = scaler_y_fit.inverse_transform(preds1.reshape(-1, 1)).flatten()
    preds2_real = scaler_y_fit.inverse_transform(preds2.reshape(-1, 1)).flatten()

    accuracy1 = np.mean(np.abs(preds1_real[:len(y_val_real)] - y_val_real) <= 0.1 * y_val_real) * 100
    accuracy2 = np.mean(np.abs(preds2_real[:len(y_val_real)] - y_val_real) <= 0.1 * y_val_real) * 100

    plt.figure(figsize=(10, 5))
    plt.bar(['BNN', 'DBNN'], [accuracy1, accuracy2], color=['blue', 'orange'])
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.title('Model Accuracy', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0, 100)
    plt.savefig('model_accuracy.png')
    plt.show()

if __name__ == "__main__":
    data_path = os.path.join(project_root, 'data', 'cleaning_data_test.csv')
    val_split = 0.2
    seed = 1601
    batch_size = 2048
    epochs = 100
    learning_rate = 1e-4

    data_scaled_x, data_scaled_y, scaler_x_fit, scaler_y_fit = load_and_preprocess_data(data_path)
    x_train, y_train, x_val, y_val = split_data(data_scaled_x, data_scaled_y, val_split, seed)

    bnn_model = train_bayesian_regression(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, "BNN", scaler_y_fit)
    dbnn_model = train_bayesian_density_network(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, val_split, "DBNN", scaler_y_fit)

    data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0])
    make_predictions_and_plot_residuals(bnn_model, dbnn_model, data_val, scaler_y_fit)
    plot_predictive_distributions(bnn_model, dbnn_model, data_val, scaler_y_fit)
    compute_coverage_and_errors(bnn_model, dbnn_model, x_val, y_val, scaler_x_fit, scaler_y_fit)
