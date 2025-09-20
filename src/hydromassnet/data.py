# file: ./src/hydromassnet/data.py
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    """
    Classe unificada para carregar, dividir e escalar os dados do projeto.
    """
    def __init__(self, config: dict, feature_override: list = None):
        """
        Inicializa o DataHandler.

        Parameters
        ----------
        config : dict
            Dicionário de configuração principal do projeto.
        feature_override : list, optional
            Uma lista de features para usar no lugar daquelas no config.
        """
        self.config = config
        self.paths = config['paths']
        self.data_cfg = config['data_processing']
        self.features = feature_override if feature_override else self.data_cfg.get('features', [])
        self.target = config['target_column']
        self.seed = config['seed']

        os.makedirs(self.paths['results'], exist_ok=True)
        self.scaler_x_path = os.path.join(self.paths['results'], 'scaler_x.pkl')
        self.scaler_y_path = os.path.join(self.paths['results'], 'scaler_y.pkl')

        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def get_full_dataset_and_splits(self) -> tuple:
        """
        Carrega, divide e escala os dados. Salva os scalers para uso futuro.

        Returns
        -------
        tuple
            (x_train, y_train, x_val, y_val, x_test, y_test, features)
        """
        df = pd.read_csv(self.paths['processed_data'])

        X = df[self.features]
        y = df[[self.target]]

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.data_cfg['test_split'],
            random_state=self.seed
        )

        val_size_adjusted = self.data_cfg['val_split'] / (1 - self.data_cfg['test_split'])

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.seed
        )

        X_train_scaled = self.scaler_x.fit_transform(X_train)
        X_val_scaled = self.scaler_x.transform(X_val)
        X_test_scaled = self.scaler_x.transform(X_test)

        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)

        joblib.dump(self.scaler_x, self.scaler_x_path)
        joblib.dump(self.scaler_y, self.scaler_y_path)

        return (
            X_train_scaled.astype('float32'), y_train_scaled.astype('float32'),
            X_val_scaled.astype('float32'), y_val_scaled.astype('float32'),
            X_test_scaled.astype('float32'), y_test_scaled.astype('float32'),
            self.features
        )
