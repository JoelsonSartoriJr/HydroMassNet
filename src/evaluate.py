import os
import argparse
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from src.data import DataHandler
import yaml

def main(args):
    # Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Garantir test_split default
    if "data_processing" not in config:
        config["data_processing"] = {}
    if "test_split" not in config["data_processing"]:
        config["data_processing"]["test_split"] = 0.2

    data_handler = DataHandler(config)
    _, _, _, _, x_test, y_test, _ = data_handler.get_full_dataset_and_splits()

    # Modelo
    if args.model == "vanilla":
        from src.models.vanilla_network import VanillaNetwork
        model = VanillaNetwork(input_shape=(x_test.shape[1],), config={"layers": "64-32"})
    elif args.model == "bnn":
        from src.models.bayesian_dense_network import BayesianDenseNetwork
        model = BayesianDenseNetwork(input_shape=(x_test.shape[1],), config={"layers": "128-64"})
    elif args.model == "dbnn":
        from src.models.bayesian_density_network import BayesianDensityNetwork
        model = BayesianDensityNetwork(input_shape=(x_test.shape[1],), config={"layers": "128-64"})
    else:
        raise ValueError(f"Modelo desconhecido: {args.model}")

    # Pesos
    model_path = os.path.join(config["paths"]["saved_models"], f"{args.model}.weights.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.load_weights(model_path)
    print(f"[✔] Pesos carregados de {model_path}")

    # Predição
    y_pred = model.predict(x_test)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_mean = np.mean(y_pred, axis=1)
        y_pred_std = np.std(y_pred, axis=1)
    else:
        y_pred_mean = y_pred.flatten()
        y_pred_std = np.zeros_like(y_pred_mean)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred_mean)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    r2 = r2_score(y_test, y_pred_mean)
    print(f"[METRICS] {args.model.upper()} -> MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

    # Salvar
    save_dir = config["paths"]["saved_models"]
    os.makedirs(save_dir, exist_ok=True)
    pred_path = os.path.join(save_dir, f"{args.model}_predictions.csv")
    pd.DataFrame({"y_true": y_test, "y_pred_mean": y_pred_mean, "y_pred_std": y_pred_std}).to_csv(pred_path, index=False)
    print(f"[✔] Predições salvas em: {pred_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Nome do modelo (bnn, dbnn, vanilla)")
    args = parser.parse_args()
    main(args)
