import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    """
    Classe para carregar, dividir e escalar os dados do projeto.
    """
    def __init__(self, config, feature_override=None):
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
        self.feature_override = feature_override
        self.scaler = MinMaxScaler()
        self.data = None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

    def load_and_split_data(self):
        """Carrega e divide os dados em treino, validação e teste."""
        df = pd.read_csv(self.config['paths']['processed_data'])

        if self.feature_override:
            features = self.feature_override
        else:
            features = self.config['data_processing']['features']

        target = self.config['target_column']

        X = df[features]
        y = df[target]

        # Primeira divisão: separar o teste
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y,
            test_size=self.config['data_processing']['test_split'],
            random_state=self.config['seed']
        )

        # Segunda divisão: separar treino e validação a partir do restante
        val_size_adjusted = self.config['data_processing']['val_split'] / (1 - self.config['data_processing']['test_split'])

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config['seed']
        )
        self.features = features

    def scale_features(self):
        """Aplica a escala MinMax nos dados de treino e transforma os de validação/teste."""
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def get_full_dataset_and_splits(self):
        """Método principal para obter todos os dados processados."""
        self.load_and_split_data()
        self.scale_features()
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.features
