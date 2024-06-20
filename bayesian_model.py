import os
import sys
import numpy as np
import tensorflow as tf
import logging
from func.data_preprocessing import load_and_preprocess_data, split_data
from func.train import train_bayesian_regression, train_bayesian_density_network
from func.plot import make_predictions_and_plot_residuals, plot_predictive_distributions, compute_coverage_and_errors, plot_accuracy_comparison, plot_error

# Suprimir avisos de "End of sequence"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# Adicionar filtro para suprimir mensagens de log específicas
class FilterOutOfRange(logging.Filter):
    def filter(self, record):
        return "OUT_OF_RANGE: End of sequence" not in record.getMessage()

logger = tf.get_logger()
logger.addFilter(FilterOutOfRange())

# Remover mensagens de warning específicas do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

def main():
    data_path = 'data/cleaning_data.csv'
    val_split = 0.2
    seed = 1601
    batch_size = 512  # Ajuste no tamanho do lote
    epochs = 500  # Ajuste no número de épocas
    learning_rate = 1e-4  # Ajuste na taxa de aprendizado
    weight_decay = 1e-6  # Adicionando weight decay

    # Carregar e pré-processar os dados
    data_scaled_x, data_scaled_y, scaler_x_fit, scaler_y_fit = load_and_preprocess_data(data_path)
    x_train, y_train, x_val, y_val = split_data(data_scaled_x, data_scaled_y, val_split, seed)

    input_shape = (None, x_train.shape[1])

    # Treinar os modelos bayesianos
    bnn_model, bnn_mae, bnn_r2 = train_bayesian_regression([13, 256, 128, 64, 32, 1], x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, weight_decay, val_split, "BNN", scaler_y_fit, input_shape)
    dbnn_model, dbnn_mae, dbnn_r2 = train_bayesian_density_network([13, 512, 256, 64], [64, 32, 1], x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, weight_decay, val_split, "DBNN", scaler_y_fit, input_shape)

    # Criar dataset de validação
    data_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(x_val.shape[0])

    # Fazer previsões e plotar resíduos
    make_predictions_and_plot_residuals(bnn_model, dbnn_model, data_val, scaler_y_fit)

    # Plotar distribuições preditivas
    plot_predictive_distributions(bnn_model, dbnn_model, data_val, scaler_y_fit)

    # Calcular cobertura e erros
    compute_coverage_and_errors(bnn_model, dbnn_model, x_val, y_val, scaler_x_fit, scaler_y_fit)

    # Plotar comparação de acurácia entre os modelos
    plot_accuracy_comparison(bnn_r2, dbnn_r2)  # Ajustado para usar r2 como proxy para accuracy

    # Plotar erro absoluto médio dos modelos
    plot_error(bnn_mae, dbnn_mae, epochs)

if __name__ == "__main__":
    main()
