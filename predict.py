import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from func.data_preprocessing import load_and_preprocess_data
from model.bayesian_dense_regression import BayesianDenseRegression
from model.bayesian_density_network import BayesianDensityNetwork

def load_models(bnn_model_path, dbnn_model_path):
    bnn_model = tf.keras.models.load_model(bnn_model_path, custom_objects={'BayesianDenseRegression': BayesianDenseRegression})
    dbnn_model = tf.keras.models.load_model(dbnn_model_path, custom_objects={'BayesianDensityNetwork': BayesianDensityNetwork})
    return bnn_model, dbnn_model

def preprocess_input(input_values, scaler_x_fit):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler_x_fit.transform(input_array)
    return input_scaled

def predict(models, input_scaled):
    bnn_model, dbnn_model = models
    bnn_prediction = bnn_model(input_scaled, training=False).numpy().flatten()
    dbnn_prediction = dbnn_model(input_scaled, training=False).numpy().flatten()
    return bnn_prediction, dbnn_prediction

def main(args):
    # Load scalers
    data_path = 'data/cleaning_data_test.csv'
    _, _, scaler_x_fit, scaler_y_fit = load_and_preprocess_data(data_path)

    # Load models
    bnn_model, dbnn_model = load_models(args.bnn_model_path, args.dbnn_model_path)

    # Preprocess input
    input_scaled = preprocess_input(args.input_values, scaler_x_fit)

    # Make predictions
    bnn_prediction, dbnn_prediction = predict((bnn_model, dbnn_model), input_scaled)

    # Inverse transform predictions
    bnn_prediction_real = scaler_y_fit.inverse_transform(bnn_prediction.reshape(-1, 1)).flatten()
    dbnn_prediction_real = scaler_y_fit.inverse_transform(dbnn_prediction.reshape(-1, 1)).flatten()

    # Print predictions
    print(f'BNN Prediction: {bnn_prediction_real[0]:.4f}')
    print(f'DBNN Prediction: {dbnn_prediction_real[0]:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using Bayesian Neural Networks')
    parser.add_argument('input_values', type=float, nargs='+', help='Input values for prediction')
    parser.add_argument('--bnn_model_path', type=str, required=True, help='Path to the saved BNN model (.h5 file)')
    parser.add_argument('--dbnn_model_path', type=str, required=True, help='Path to the saved DBNN model (.h5 file)')

    args = parser.parse_args()
    main(args)
