import pandas as pd
import yaml
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Classe para carregar, pré-processar e servir os dados como tf.data.Dataset.
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
        logger.info("DataHandler inicializado.")

    def load_and_prepare_data(self, features: list, batch_size: int):
        """
        Carrega, divide, normaliza e converte os dados para tf.data.Dataset.
        """
        logger.info(f"Carregando dados de {self.data_path} com features: {features}")
        df = pd.read_csv(self.data_path)
        X = df[features].astype('float32')
        y = df[[self.target]].astype('float32')

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_split, random_state=self.seed
        )
        val_size_adjusted = self.val_split / (1.0 - self.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=self.seed
        )

        logger.info("Normalizando os dados...")
        X_train = self.scaler_x.fit_transform(X_train)
        X_val = self.scaler_x.transform(X_val)
        X_test = self.scaler_x.transform(X_test)
        y_train = self.scaler_y.fit_transform(y_train)
        y_val = self.scaler_y.transform(y_val)
        y_test = self.scaler_y.transform(y_test)
        logger.info("Dados normalizados.")

        logger.info(f"Criando datasets com batch size = {batch_size}")
        train_dataset = self._to_tf_dataset(X_train, y_train, batch_size)
        val_dataset = self._to_tf_dataset(X_val, y_val, batch_size)
        test_dataset = self._to_tf_dataset(X_test, y_test, batch_size, shuffle=False)

        self.n_features = X_train.shape[1]
        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def _to_tf_dataset(x, y, batch_size, shuffle=True):
        """Cria um tf.data.Dataset otimizado."""
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(x))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset