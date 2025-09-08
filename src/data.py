# file: ./src/data.py
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:
    """
    Classe para carregar, pré-processar e dividir os dados.
    """
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.data_path = self.config['paths']['processed_data']
        self.target = self.config['target_column']
        self.seed = self.config['seed']
        self.val_split = self.config['data_processing']['val_split']
        self.test_split = self.config['data_processing']['test_split']
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

    def load_and_prepare_data(self, features: list):
        """
        Carrega os dados, divide em conjuntos de treino/validação/teste e aplica a normalização.

        Args:
            features (list): Lista de colunas de features a serem usadas.

        Returns:
            Tuple: Contendo X_train, y_train, X_val, y_val, X_test, y_test.
        """
        df = pd.read_csv(self.data_path)
        X = df[features]
        y = df[[self.target]]

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=self.seed
        )

        val_size_adjusted = self.val_split / (1.0 - self.test_split)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=self.seed
        )

        X_train = self.scaler_x.fit_transform(X_train)
        X_val = self.scaler_x.transform(X_val)
        X_test = self.scaler_x.transform(X_test)

        y_train = self.scaler_y.fit_transform(y_train)
        y_val = self.scaler_y.transform(y_val)
        y_test = self.scaler_y.transform(y_test)

        return X_train, y_train, X_val, y_val, X_test, y_test
