import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score

def make_predictions_and_plot_residuals(model1, model2, data_val, scaler_y_fit):
    resids1 = []
    resids2 = []
    for x_data, y_data in data_val:
        resids1.append(y_data[:, 0] - model1(x_data, training=False)[:, 0])
        resids2.append(y_data[:, 0] - model2(x_data, training=False)[:, 0])

    resids1 = np.concatenate(resids1)
    resids2 = np.concatenate(resids2)

    bins = np.linspace(-2, 2, 100)
    plt.hist(resids1, bins, alpha=0.5, label='No Error Estimation')
    plt.hist(resids2, bins, alpha=0.5, label='Density Network')
    plt.legend()
    plt.xlabel('Residuals', fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig('Residuo.png')

def plot_predictive_distributions(model1, model2, data_val, scaler_y_fit):
    samples1 = []
    samples2 = []
    for x_data, y_data in data_val:
        samples1.append(model1(x_data, training=False))
        samples2.append(model2(x_data, training=False))

    samples1 = np.concatenate(samples1, axis=0)
    samples2 = np.concatenate(samples2, axis=0)
    y_data_real = scaler_y_fit.inverse_transform(y_data)

    plt.figure(figsize=(20, 10))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        num = np.random.randint(len(y_data_real), size=1)[0]
        samples1_real = scaler_y_fit.inverse_transform(samples1[num, :].reshape(1, -1))[0]
        samples2_real = scaler_y_fit.inverse_transform(samples2[num, :].reshape(1, -1))[0]
        sns.kdeplot(samples1_real, fill=True, label='BNN')
        sns.kdeplot(samples2_real, fill=True, label='DBNN', warn_singular=False)
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
    q0 = (100.0 - prc) / 2.0
    q1 = 100.0 - q0
    within_conf_int = np.zeros(len(y_true))
    for i in range(len(y_true)):
        p0 = np.percentile(samples[i, :], q0)
        p1 = np.percentile(samples[i, :], q1)
        if p0 <= y_true[i] and p1 > y_true[i]:
            within_conf_int[i] = 1
    return within_conf_int

def compute_coverage_and_errors(model1, model2, x_val, y_val, scaler_x_fit, scaler_y_fit):
    samples1 = model1(x_val, training=False)
    samples2 = model2(x_val, training=False)

    covered1 = covered(samples1, y_val)
    covered2 = covered(samples2, y_val)

    mean1 = np.mean(samples1, axis=1)
    mean2 = np.mean(samples2, axis=1)

    print(f'Erro médio absoluto BNN {mean_absolute_error(y_val, mean1):.3f}')
    print(f'Score R2 BNN: {r2_score(y_val, mean1) * 100:.3f}')

    print(f'Erro médio absoluto DBNN {mean_absolute_error(y_val, mean2):.3f}')
    print(f'Score R2 DBNN: {r2_score(y_val, mean2) * 100:.3f}')

    mean1 = mean1.reshape(-1, 1)
    mean2 = mean2.reshape(-1, 1)
    x_val_real = scaler_x_fit.inverse_transform(x_val)

    star_formation = x_val_real[:, -1]

    preds_real = scaler_y_fit.inverse_transform(mean1).flatten()
    y_val_real = scaler_y_fit.inverse_transform(y_val).flatten()

    plt.figure(figsize=(32, 7))

    error = np.abs(preds_real - y_val_real)
    plt.plot(y_val_real, y_val_real, color='green', zorder=3)
    plt.errorbar(y_val_real, preds_real, yerr=error, fmt="o", alpha=0.7)
    plt.scatter(y_val_real, preds_real, c=star_formation, zorder=3, alpha=0.7)
    plt.xlabel(r'True values ($log(M_{\odot})$)', fontsize=24)
    plt.ylabel(r'Predict values ($log(M_{\odot})$)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim(8.5, 10.7)
    plt.ylim(7, 12)
    plt.savefig('realVSpreditoBNN.png')
    plt.show()

    preds_real = scaler_y_fit.inverse_transform(mean2).flatten()
    y_val_real = scaler_y_fit.inverse_transform(y_val).flatten()

    plt.figure(figsize=(32, 7))

    error = np.abs(preds_real - y_val_real)
    plt.plot(y_val_real, y_val_real, color='green', zorder=3)
    plt.errorbar(y_val_real, preds_real, yerr=error, fmt="o", alpha=0.7)
    plt.scatter(y_val_real, preds_real, c=star_formation, zorder=3, alpha=0.7)
    plt.xlabel(r'True values ($log(M_{\odot})$)', fontsize=24)
    plt.ylabel(r'Predict values ($log(M_{\odot})$)', fontsize=24)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlim(8.5, 10.7)
    plt.ylim(7, 12)
    plt.savefig('realVSpreditoDBNN.png')
    plt.show()

def plot_accuracy(model1, model2, x_val, y_val, scaler_y_fit):
    preds1 = model1(x_val, training=False).numpy().flatten()
    preds2 = model2(x_val, training=False).numpy().flatten()

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
