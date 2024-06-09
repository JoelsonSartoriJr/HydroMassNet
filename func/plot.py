import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(elbo, mae, r2, model_name):
    epochs = range(1, len(elbo) + 1)

    plt.figure(figsize=(18, 5))

    # Plot ELBO
    plt.subplot(1, 3, 1)
    plt.plot(epochs, elbo, 'b', label='ELBO')
    plt.title(f'ELBO during training for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('ELBO')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, mae, 'r', label='MAE')
    plt.title(f'MAE during training for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    # Plot R²
    plt.subplot(1, 3, 3)
    plt.plot(epochs, r2, 'g', label='R²')
    plt.title(f'R² during training for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()

    plt.tight_layout()
    plt.show()

def make_predictions_and_plot_residuals(model1, model2, data_val, scaler_y_fit):
    for x_data, y_data in data_val:
        preds1 = model1(x_data, training=False).numpy()
        preds2 = model2(x_data, training=False).numpy()

        preds1_real = scaler_y_fit.inverse_transform(preds1[:, 0].reshape(-1, 1))
        preds2_real = scaler_y_fit.inverse_transform(preds2[:, 0].reshape(-1, 1))
        y_val_real = scaler_y_fit.inverse_transform(y_data.numpy().reshape(-1, 1))

        resids1 = y_val_real - preds1_real
        resids2 = y_val_real - preds2_real

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(y_val_real, resids1, alpha=0.5, label='BNN')
        plt.hlines(0, min(y_val_real), max(y_val_real), colors='r')
        plt.xlabel('True values')
        plt.ylabel('Residuals')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(y_val_real, resids2, alpha=0.5, label='DBNN')
        plt.hlines(0, min(y_val_real), max(y_val_real), colors='r')
        plt.xlabel('True values')
        plt.ylabel('Residuals')
        plt.legend()

        plt.suptitle('Residuals')
        plt.tight_layout()
        plt.show()

def plot_predictive_distributions(model1, model2, data_val, scaler_y_fit):
    for x_data, y_data in data_val:
        preds1 = model1(x_data, training=False).numpy()
        preds2 = model2(x_data, training=False).numpy()

        preds1_real = scaler_y_fit.inverse_transform(preds1[:, 0].reshape(-1, 1))
        preds2_real = scaler_y_fit.inverse_transform(preds2[:, 0].reshape(-1, 1))
        y_val_real = scaler_y_fit.inverse_transform(y_data.numpy().reshape(-1, 1))

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(preds1_real, bins=50, alpha=0.5, label='BNN', density=True)
        plt.hist(y_val_real, bins=50, alpha=0.5, label='True', density=True)
        plt.xlabel('Predicted values')
        plt.ylabel('Density')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(preds2_real, bins=50, alpha=0.5, label='DBNN', density=True)
        plt.hist(y_val_real, bins=50, alpha=0.5, label='True', density=True)
        plt.xlabel('Predicted values')
        plt.ylabel('Density')
        plt.legend()

        plt.suptitle('Predictive Distributions')
        plt.tight_layout()
        plt.show()

def compute_coverage_and_errors(model1, model2, x_val, y_val, scaler_x_fit, scaler_y_fit):
    y_pred1 = model1(x_val, sampling=True).numpy()
    y_pred2 = model2(x_val, sampling=True).numpy()

    y_pred1_real = scaler_y_fit.inverse_transform(y_pred1[:, 0].reshape(-1, 1))
    y_pred2_real = scaler_y_fit.inverse_transform(y_pred2[:, 0].reshape(-1, 1))
    y_val_real = scaler_y_fit.inverse_transform(y_val.reshape(-1, 1))

    coverage1 = np.mean((np.abs(y_pred1_real - y_val_real) / y_val_real) <= 0.1) * 100
    coverage2 = np.mean((np.abs(y_pred2_real - y_val_real) / y_val_real) <= 0.1) * 100

    print(f'Coverage for BNN: {coverage1:.2f}%')
    print(f'Coverage for DBNN: {coverage2:.2f}%')

def plot_accuracy(bnn_model, dbnn_model, x_val, y_val, scaler_y_fit):
    y_pred1 = bnn_model(x_val, sampling=False).numpy()
    y_pred2 = dbnn_model(x_val, sampling=False).numpy()

    y_pred1_real = scaler_y_fit.inverse_transform(y_pred1[:, 0].reshape(-1, 1))
    y_pred2_real = scaler_y_fit.inverse_transform(y_pred2[:, 0].reshape(-1, 1))
    y_val_real = scaler_y_fit.inverse_transform(y_val.reshape(-1, 1))

    accuracy1 = np.mean((np.abs(y_pred1_real - y_val_real) / y_val_real) <= 0.1) * 100
    accuracy2 = np.mean((np.abs(y_pred2_real - y_val_real) / y_val_real) <= 0.1) * 100

    plt.figure(figsize=(12, 6))
    plt.bar(['BNN', 'DBNN'], [accuracy1, accuracy2])
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison between BNN and DBNN')
    plt.ylim([0, 100])
    plt.show()

def plot_error(mae1, mae2, epochs):
    epochs_range = range(1, epochs + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs_range, mae1, label='BNN MAE')
    plt.plot(epochs_range, mae2, label='DBNN MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE Comparison between BNN and DBNN')
    plt.show()
