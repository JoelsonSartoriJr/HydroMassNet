import os
import tensorflow as tf
import pandas as pd
from src.data import DataHandler

def train(model_name, features, exp_config, main_config):
    """
    Treina um modelo dado seu nome e configuração.
    Retorna um DataFrame com histórico de treinamento.
    """
    print(f"[INFO] Iniciando treino do modelo: {model_name.upper()}")

    # Dados
    data_handler = DataHandler(main_config)
    x_train, y_train, x_val, y_val, _, _, _ = data_handler.get_full_dataset_and_splits()

    # Modelo
    if model_name == "vanilla":
        from src.models.vanilla_network import VanillaNetwork
        model = VanillaNetwork(input_shape=(x_train.shape[1],), config=exp_config)
    elif model_name == "bnn":
        from src.models.bayesian_dense_network import BayesianDenseNetwork
        model = BayesianDenseNetwork(input_shape=(x_train.shape[1],), config=exp_config)
    elif model_name == "dbnn":
        from src.models.bayesian_density_network import BayesianDensityNetwork
        model = BayesianDensityNetwork(input_shape=(x_train.shape[1],), config=exp_config)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    # Compilar
    optimizer = tf.keras.optimizers.Adam(learning_rate=exp_config.get("learning_rate", 1e-3))
    model.compile(optimizer=optimizer)

    # Treinar
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=exp_config.get("epochs", 50),
        batch_size=exp_config.get("batch_size", 32),
        verbose=1,
    )

    # Salvar modelo
    save_dir = main_config["paths"]["saved_models"]
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.weights.h5")
    model.save_weights(model_path)
    print(f"[✔] Pesos salvos em: {model_path}")

    # Histórico
    hist_df = pd.DataFrame(history.history)
    hist_path = os.path.join(save_dir, f"{model_name}_history.csv")
    hist_df.to_csv(hist_path, index=False)
    print(f"[✔] Histórico salvo em: {hist_path}")

    return hist_df
