import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['u-r'] = data['umag'] - data['rmag']
    data = data.sample(frac=1)
    train_labels = ['umag', 'gmag', 'rmag', 'imag', 'g-r', 'u-r', 'Vhelio',
                    'expAB_r', 'gamma_g', 'gamma_i', 'gmi_corr',
                    'logMstarMcGaugh', 'logSFR22']

    data_x = data[train_labels]
    data_y = data[['logMH']]

    scaler_x = MinMaxScaler()
    scaler_x_fit = scaler_x.fit(data_x)
    data_scaled_x = scaler_x_fit.transform(data_x)
    data_scaled_x = pd.DataFrame(data_scaled_x.astype('float32'), columns=train_labels)

    scaler_y = MinMaxScaler()
    scaler_y_fit = scaler_y.fit(data_y)
    data_scaled_y = scaler_y_fit.transform(data_y)
    data_scaled_y = pd.DataFrame(data_scaled_y.astype('float32'), columns=['logMH'])
    data_scaled_y = data_scaled_y['logMH']

    return data_scaled_x, data_scaled_y, scaler_x_fit, scaler_y_fit

def split_data(data_scaled_x, data_scaled_y, val_split, seed):
    np.random.seed(seed)
    tr_ind = np.random.choice([False, True],
                              size=data_scaled_x.shape[0],
                              p=[val_split, 1.0-val_split])
    x_train = data_scaled_x[tr_ind].values
    y_train = data_scaled_y[tr_ind].values
    x_val = data_scaled_x[~tr_ind].values
    y_val = data_scaled_y[~tr_ind].values
    y_train = np.expand_dims(y_train, 1)
    y_val = np.expand_dims(y_val, 1)

    return x_train, y_train, x_val, y_val
