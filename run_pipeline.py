# file: ./run_pipeline.py
import argparse

from src.hydromassnet.plotting import plot_all
from src.hydromassnet.preprocess import clean_and_feature_engineer
from src.pipelines import run_dataset_pipeline


def main(config_path: str = "config.yaml"):
    """Orquestra o pipeline completo de pré-processamento, treino e avaliação."""
    run_dataset_pipeline(
        config_path=config_path,
        preprocess_fn=clean_and_feature_engineer,
        performance_plotter=plot_all,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline completo HydroMassNet")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Caminho para o arquivo de configuração (default: config.yaml).",
    )
    args = parser.parse_args()
    main(args.config)
