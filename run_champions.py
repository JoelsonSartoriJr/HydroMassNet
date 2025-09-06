import os
import yaml
import importlib
import pandas as pd
from datetime import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suprimir logs TensorFlow

def train_wrapper(model_type, features, exp_config, main_config):
    train_module = importlib.import_module("src.train")
    hist_df = train_module.train(model_type, features, exp_config, main_config)
    if not isinstance(hist_df, pd.DataFrame):
        raise RuntimeError("train() não retornou DataFrame.")
    return hist_df

def main():
    with open("champion_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    for champion_name, champion_cfg in config["champions"].items():
        model_type = champion_cfg["model"]
        features = champion_cfg.get("features", [])
        exp_config = champion_cfg.get("hyperparams", {})
        main_config = config

        print("\n" + "#"*80)
        print(f"### Rodando {champion_name.upper()} ###")
        print("#"*80)

        try:
            history_df = train_wrapper(model_type, features, exp_config, main_config)
            save_dir = os.path.join("results", "champions", datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(save_dir, exist_ok=True)
            hist_path = os.path.join(save_dir, f"{champion_name}_history.csv")
            history_df.to_csv(hist_path, index=False)
            print(f"[✔] Histórico salvo em {hist_path}")
        except Exception as e:
            print(f"[!] Erro durante treino de {champion_name}: {e}")

        # Avaliar modelo
        try:
            os.system(f"poetry run python -m src.evaluate --model {model_type}")
        except Exception as e:
            print(f"[!] Avaliação falhou para {champion_name}: {e}")

if __name__ == "__main__":
    main()
