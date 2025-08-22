import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.feature_labels = self.config['data_processing']['features']

    def load_and_preprocess(self, training=True):
        data_path = self.config['paths']['data'] if training else self.config['paths']['test_data']
        data = pd.read_csv(data_path)

        # Engenharia de feature
        if 'u-r' in self.feature_labels:
            data['u-r'] = data['umag'] - data['rmag']

        data = data.sample(frac=1, random_state=self.config['seed'])

        data_x = data[self.feature_labels]
        data_y = data[[self.config['target_column']]]

        # Fit e transform para dados de treino, apenas transform para predição
        if training:
            data_scaled_x = self.scaler_x.fit_transform(data_x)
            data_scaled_y = self.scaler_y.fit_transform(data_y)
            # Salvar scalers
            os.makedirs(self.config['paths']['saved_models'], exist_ok=True)
            joblib.dump(self.scaler_x, os.path.join(self.config['paths']['saved_models'], 'scaler_x.pkl'))
            joblib.dump(self.scaler_y, os.path.join(self.config['paths']['saved_models'], 'scaler_y.pkl'))
        else:
            self.scaler_x = joblib.load(os.path.join(self.config['paths']['saved_models'], 'scaler_x.pkl'))
            self.scaler_y = joblib.load(os.path.join(self.config['paths']['saved_models'], 'scaler_y.pkl'))
            data_scaled_x = self.scaler_x.transform(data_x)
            data_scaled_y = self.scaler_y.transform(data_y)

        return data_scaled_x, data_scaled_y.flatten()

    def split_data(self, data_x, data_y):
        np.random.seed(self.config['seed'])
        val_split = self.config['data_processing']['val_split']
        train_idx = np.random.choice([False, True], size=data_x.shape[0], p=[val_split, 1.0 - val_split])

        x_train, y_train = data_x[train_idx], data_y[train_idx]
        x_val, y_val = data_x[~train_idx], data_y[~train_idx]

        return x_train, np.expand_dims(y_train, 1), x_val, np.expand_dims(y_val, 1)
