import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class DataHandler:
    # Adicionamos 'feature_override' ao construtor
    def __init__(self, config: dict, feature_override: list = None):
        self.config = config
        self.scaler_x_path = os.path.join(self.config['paths']['saved_models'], 'scaler_x.pkl')
        self.scaler_y_path = os.path.join(self.config['paths']['saved_models'], 'scaler_y.pkl')
        # Usa a lista de override se fornecida, senão, usa a do config
        self.feature_labels = feature_override if feature_override else self.config['data_processing']['features']
        self.target_column = self.config['target_column']
        self.seed = self.config['seed']

    # O resto da classe permanece o mesmo...
    def _load_data(self) -> pd.DataFrame:
        """Carrega os dados processados."""
        data = pd.read_csv(self.config['paths']['processed_data'])
        return data.sample(frac=1, random_state=self.seed).reset_index(drop=True)

    def get_full_dataset_and_splits(self) -> tuple:
        """
        Carrega, pré-processa e divide o conjunto de dados completo.
        """
        data = self._load_data()

        # Usa self.feature_labels que já foi definido no __init__
        features = [feat for feat in self.feature_labels if feat in data.columns]

        data_x = data[features]
        data_y = data[[self.target_column]]

        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()

        data_scaled_x = scaler_x.fit_transform(data_x).astype(np.float32)
        data_scaled_y = scaler_y.fit_transform(data_y).astype(np.float32)

        os.makedirs(os.path.dirname(self.scaler_x_path), exist_ok=True)
        joblib.dump(scaler_x, self.scaler_x_path)
        joblib.dump(scaler_y, self.scaler_y_path)

        val_split = self.config['data_processing']['val_split']
        test_split = self.config['data_processing']['test_split']
        dataset_size = data_scaled_x.shape[0]
        indices = np.random.permutation(dataset_size)

        test_split_point = int(dataset_size * test_split)
        val_split_point = test_split_point + int(dataset_size * val_split)

        test_idx = indices[:test_split_point]
        val_idx = indices[test_split_point:val_split_point]
        train_idx = indices[val_split_point:]

        x_train, y_train = data_scaled_x[train_idx], data_scaled_y[train_idx]
        x_val, y_val = data_scaled_x[val_idx], data_scaled_y[val_idx]
        x_test, y_test = data_scaled_x[test_idx], data_scaled_y[test_idx]

        return x_train, y_train, x_val, y_val, x_test, y_test, features
