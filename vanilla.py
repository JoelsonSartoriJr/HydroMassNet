import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.metrics import r2_score, mean_absolute_error
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

SEED = 1601
set_seed(SEED)

sns.set()
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (24, 12)
plt.rcParams["font.size"] = 18

class LrHistory(Callback):
    def __init__(self, lr_history):
        super().__init__()
        self.lr_history = lr_history

    def on_epoch_end(self, epoch, logs=None):
        self.lr_history.append(float(tf.keras.backend.get_value(self.model.optimizer.learning_rate)))

class VanillaNeuralNetwork:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.train_labels = ['umag', 'gmag', 'rmag', 'imag', 'g-r', 'u-r', 'Vhelio',
                             'expAB_r', 'gamma_g', 'gamma_i', 'gmi_corr',
                             'logMstarMcGaugh', 'logSFR22']
        self.args = {
            "batch_size": 1024,
            "epochs": 500,
            "lr": 2e-3,
            "val_split": 0.2
        }
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.model = None
        self.history = None
        self.lr_history = []

    def load_and_preprocess_data(self):
        data = pd.read_csv(self.data_path)
        data['u-r'] = data['umag'] - data['rmag']
        data = data.sample(frac=1)
        data_x = data[self.train_labels]
        data_y = data[['logMH']]

        data_scaled_x = self.scaler_x.fit_transform(data_x)
        data_scaled_y = self.scaler_y.fit_transform(data_y)

        data_scaled_x = pd.DataFrame(data_scaled_x.astype('float32'), columns=self.train_labels)
        data_scaled_y = pd.DataFrame(data_scaled_y.astype('float32'), columns=['logMH'])

        train_idx = np.random.choice(
            [False, True],
            size=data_scaled_x.shape[0],
            p=[self.args['val_split'], 1.0 - self.args['val_split']]
        )

        self.X_train = data_scaled_x[train_idx].values
        self.y_train = data_scaled_y[train_idx].values
        self.X_val = data_scaled_x[~train_idx].values
        self.y_val = data_scaled_y[~train_idx].values

    def build_model(self, input_shape: tuple):
        self.model = Sequential([
            Input(shape=(input_shape,)),
            Dense(64, use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),
            Dense(128, use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),
            Dense(64, use_bias=False),
            BatchNormalization(),
            ReLU(),
            Dropout(0.2),
            Dense(1)
        ])
        self.model.compile(Adam(learning_rate=self.args['lr']), loss='mean_absolute_error', metrics=['accuracy'])

    def train_model(self):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=10, min_lr=0.00001)

        lr_history_callback = LrHistory(self.lr_history)

        self.history = self.model.fit(self.X_train, self.y_train,
                                      batch_size=self.args['batch_size'],
                                      epochs=self.args['epochs'],
                                      validation_split=self.args['val_split'],
                                      callbacks=[reduce_lr, lr_history_callback],
                                      verbose=0)

    def plot_loss(self):
        plt.figure(figsize=(15, 10))
        plt.plot(self.history.history['loss'], label='Train')
        plt.plot(self.history.history['val_loss'], label='Val')
        plt.legend(prop={"size": 22})
        plt.xlabel('Epoch', fontsize=22)
        plt.xticks(fontsize=22)
        plt.ylabel('MAE', fontsize=22)
        plt.yticks(fontsize=22)
        plt.savefig('loss.png')
        plt.show()

        plt.figure(figsize=(13, 8))
        plt.plot(self.lr_history)
        plt.xlabel('Epoch')
        plt.ylabel('Learning rate')
        plt.show()

    def plot_predictions(self):
        preds = self.model.predict(self.X_val)
        preds_real = self.scaler_y.inverse_transform(preds)
        y_val_real = self.scaler_y.inverse_transform(self.y_val)

        plt.figure(figsize=(18, 10))
        for i in range(16):
            sample = np.random.randint(len(preds_real), size=1)
            plt.subplot(4, 4, i + 1)
            plt.axvline(preds_real[sample[0]], label='Pred', linewidth=3)
            plt.axvline(y_val_real[sample[0]], ls=':', color='red', label='True', linewidth=3)
            plt.gca().get_yaxis().set_ticklabels([])
            plt.xticks(fontsize=16)
            if i < 12:
                plt.gca().get_xaxis().set_ticklabels([])
            if i == 3:
                plt.legend()
        plt.savefig('Pred.png')
        plt.show()

    def plot_real_vs_predicted(self):
        preds = self.model.predict(self.X_val)
        preds_real = self.scaler_y.inverse_transform(preds)
        y_val_real = self.scaler_y.inverse_transform(self.y_val)
        x_val_real = self.scaler_x.inverse_transform(self.X_val)

        star_formation = x_val_real[:, -1]

        plt.figure(figsize=(32, 7))

        error = np.abs(preds_real - y_val_real).flatten()

        plt.plot(y_val_real, y_val_real, color='green', zorder=3)
        plt.errorbar(y_val_real.flatten(), preds_real.flatten(), yerr=error, fmt="o", alpha=0.7)
        plt.scatter(y_val_real.flatten(), preds_real.flatten(), c=star_formation, zorder=3, alpha=0.7)

        plt.xlabel(r'True values ($log(M_{\odot} )$)', fontsize=24)
        plt.ylabel(r'Predict values ($log(M_{\odot})$)', fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)

        plt.xlim(8.5, 10.7)
        plt.ylim(7, 12)

        plt.savefig('realVSpreditoNN.png')
        plt.show()


if __name__ == "__main__":
    data_path = 'data/cleaning_data_test.csv'
    vnn = VanillaNeuralNetwork(data_path)
    vnn.load_and_preprocess_data()
    vnn.build_model(vnn.X_train.shape[1])
    vnn.train_model()
    vnn.plot_loss()
    vnn.plot_predictions()
    vnn.plot_real_vs_predicted()
