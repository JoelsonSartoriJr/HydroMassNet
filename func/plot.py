import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(elbo, mae, r2, model_name):
    epochs = range(1, len(elbo) + 1)

    # Plot ELBO
    plt.figure()
    plt.plot(epochs, elbo, 'b', label='ELBO')
    plt.title(f'ELBO during training for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('ELBO')
    plt.legend()
    plt.savefig(f'{model_name}_ELBO.png')
    plt.show()

    # Plot MAE
    plt.figure()
    plt.plot(epochs, mae, 'r', label='MAE')
    plt.title(f'MAE during training for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(f'{model_name}_MAE.png')
    plt.show()

    # Plot R²
    plt.figure()
    plt.plot(epochs, r2, 'g', label='R²')
    plt.title(f'R² during training for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('R²')
    plt.legend()
    plt.savefig(f'{model_name}_R2.png')
    plt.show()

def plot_accuracy(accuracy, model_name):
    epochs = range(1, len(accuracy) + 1)

    plt.figure()
    plt.plot(epochs, accuracy, label=f'Accuracy {model_name}')
    plt.title(f'Accuracy during training for {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(f'{model_name}_Accuracy.png')
    plt.show()

def plot_accuracy_comparison(accuracy_bnn, accuracy_dbnn):
    epochs = range(1, len(accuracy_bnn) + 1)

    plt.figure()
    plt.plot(epochs, accuracy_bnn, label='BNN Accuracy', color='blue')
    plt.plot(epochs, accuracy_dbnn, label='DBNN Accuracy', color='green')
    plt.title('Accuracy Comparison between BNN and DBNN')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('Accuracy_Comparison.png')
    plt.show()

def make_predictions_and_plot_residuals(model1, model2, data_val, scaler_y_fit):
    for x_data, y_data in data_val:
        preds1 = model1(x_data, training=False).numpy()
        preds2 = model2(x_data, training=False).numpy()

        preds1_real = scaler_y_fit.inverse_transform(preds1[:, 0].reshape(-1, 1))
        preds2_real = scaler_y_fit.inverse_transform(preds2[:, 0].reshape(-1, 1))
        y_val_real = scaler_y_fit.inverse_transform(y_data.numpy().reshape(-1, 1))

        resids1 = (preds1_real - y_val_real).flatten()
        resids2 = (preds2_real - y_val_real).flatten()

        plt.figure(figsize=(16, 7))

        plt.subplot(1, 2, 1)
        plt.plot(y_val_real.flatten(), y_val_real.flatten(), color='green', zorder=3)
        plt.errorbar(y_val_real.flatten(), preds1_real.flatten(), yerr=np.abs(resids1), fmt="o", alpha=0.7, markersize=5, capsize=3, color='blue', ecolor='lightgray', elinewidth=2)
        plt.scatter(y_val_real.flatten(), preds1_real.flatten(), zorder=3, alpha=0.7, color='blue')
        plt.xlabel(r'True values ($log(M_{\odot})$)', fontsize=24)
        plt.ylabel(r'Predict values ($log(M_{\odot})$)', fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlim(8.5, 10.7)
        plt.ylim(7, 12)
        plt.title('True vs Predicted for BNN')
        plt.grid(True)
        plt.legend(['Ideal', 'Predictions'], fontsize=16)
        plt.savefig('realVSpreditoBNN.png')

        plt.subplot(1, 2, 2)
        plt.plot(y_val_real.flatten(), y_val_real.flatten(), color='green', zorder=3)
        plt.errorbar(y_val_real.flatten(), preds2_real.flatten(), yerr=np.abs(resids2), fmt="o", alpha=0.7, markersize=5, capsize=3, color='red', ecolor='lightgray', elinewidth=2)
        plt.scatter(y_val_real.flatten(), preds2_real.flatten(), zorder=3, alpha=0.7, color='red')
        plt.xlabel(r'True values ($log(M_{\odot})$)', fontsize=24)
        plt.ylabel(r'Predict values ($log(M_{\odot})$)', fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.xlim(8.5, 10.7)
        plt.ylim(7, 12)
        plt.title('True vs Predicted for DBNN')
        plt.grid(True)
        plt.legend(['Ideal', 'Predictions'], fontsize=16)
        plt.savefig('realVSpreditoDBNN.png')

        plt.suptitle('True vs Predicted Values and Error Rates', fontsize=26)
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
        plt.hist(preds1_real, bins=50, alpha=0.5, label='BNN', density=True, color='blue')
        plt.hist(y_val_real, bins=50, alpha=0.5, label='True', density=True, color='red')
        plt.xlabel('Predicted values')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Predictive Distribution for BNN')
        plt.savefig('Predictive_Distribution_BNN.png')

        plt.subplot(1, 2, 2)
        plt.hist(preds2_real, bins=50, alpha=0.5, label='DBNN', density=True, color='green')
        plt.hist(y_val_real, bins=50, alpha=0.5, label='True', density=True, color='red')
        plt.xlabel('Predicted values')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Predictive Distribution for DBNN')
        plt.savefig('Predictive_Distribution_DBNN.png')

        plt.suptitle('Predictive Distributions', fontsize=20)
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

def plot_error(mae1, mae2, epochs):
    epochs_range = range(1, epochs + 1)

    plt.figure()
    plt.plot(epochs_range, mae1, label='BNN MAE', color='blue')
    plt.plot(epochs_range, mae2, label='DBNN MAE', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE Comparison between BNN and DBNN')
    plt.savefig('MAE_Comparison.png')
    plt.show()
