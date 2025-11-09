# file: ./main_xgass.py
import argparse

from src.hydromassnet.plotting import plot_all
from src.pipelines import run_dataset_pipeline
from src.xgass import (
    generate_xgass_figures,
    optimize_xgass_hyperparameters,
    prepare_xgass_dataset,
)


def main(config_path: str = "config_xgass.yaml") -> None:
    optimized_config_path = optimize_xgass_hyperparameters(config_path)
    run_dataset_pipeline(
        config_path=optimized_config_path,
        preprocess_fn=prepare_xgass_dataset,
        performance_plotter=plot_all,
        eda_plotter=generate_xgass_figures,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline para o dataset xGASS")
    parser.add_argument(
        "--config",
        type=str,
        default="config_xgass.yaml",
        help="Caminho para o arquivo de configuração (default: config_xgass.yaml).",
    )
    args = parser.parse_args()
    main(args.config)
