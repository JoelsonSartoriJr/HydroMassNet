import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1601)

class BaselineModel:
    def __init__(self, data_path: str, target_column: str):
        self.data_path = data_path
        self.target_column = target_column
        self.data = None
        self.scaled_data = None
        self.X = None
        self.y = None
        self.mae_scorer = make_scorer(mean_absolute_error)
        self.scaler = MinMaxScaler()

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.data['u-r'] = self.data['umag'] - self.data['rmag']

    def preprocess_data(self, feature_columns: list):
        self.data = self.data[feature_columns]
        self.scaled_data = self.scaler.fit_transform(self.data)
        self.scaled_data = pd.DataFrame(self.scaled_data.astype('float32'), columns=feature_columns)
        self.scaled_data = self.scaled_data.astype('float32')
        self.scaled_data = self.scaled_data.sample(frac=1)
        self.X = self.scaled_data.drop(self.target_column, axis=1)
        self.y = self.scaled_data[self.target_column]

    def cv_mae(self, regressor, cv:int=5):
        """ Prints the cross-validation mean absolute error """
        scores = cross_val_score(regressor, self.X, self.y, cv=cv, scoring=self.mae_scorer)
        print(f'MAE: {scores.mean():.3f}')
        print()

if __name__ == '__main__':
    # Usage
    feature_columns = ['umag', 'gmag', 'rmag', 'imag', 'g-r', 'u-r', 'Vhelio',
                       'expAB_r', 'gamma_g', 'gamma_i', 'gmi_corr',
                       'logMstarMcGaugh', 'logSFR22', 'logMH']

    data_path = 'data/cleaning_data_test.csv'
    target_column = 'logMH'

    baseline = BaselineModel(data_path, target_column)
    baseline.load_data()
    baseline.preprocess_data(feature_columns)

    # Dummy Regressor
    baseline.cv_mae(DummyRegressor())

    # CatBoost Regressor
    baseline.cv_mae(CatBoostRegressor(verbose=False, depth=9))
