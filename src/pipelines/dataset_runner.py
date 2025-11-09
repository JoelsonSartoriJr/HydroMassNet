# file: ./src/pipelines/dataset_runner.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
import yaml

from src.utils.commands import run_command

PreprocessFn = Callable[[dict], Optional[pd.DataFrame]]
EdaFn = Callable[[pd.DataFrame, dict], None]
PlotFn = Callable[[Dict[str, pd.DataFrame], dict], None]


def _load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _ensure_directories(config: dict) -> None:
    paths = config.get("paths", {})
    for key in ("results", "plots", "models"):
        if key in paths:
            os.makedirs(paths[key], exist_ok=True)


def run_dataset_pipeline(
    *,
    config_path: str,
    preprocess_fn: Optional[PreprocessFn] = None,
    performance_plotter: Optional[PlotFn] = None,
    eda_plotter: Optional[EdaFn] = None,
) -> None:
    """
    Executa o pipeline completo (pré-processamento, treino, avaliação e plots)
    para um dataset específico descrito pelo arquivo de configuração.
    """
    print(f"\n=== Iniciando pipeline para configuração: {config_path} ===")
    config = _load_config(config_path)
    _ensure_directories(config)

    processed_path = Path(config["paths"]["processed_data"])
    processed_df: Optional[pd.DataFrame] = None

    if not processed_path.exists():
        if preprocess_fn is None:
            raise FileNotFoundError(
                f"Arquivo de dados processados '{processed_path}' não encontrado "
                "e nenhuma função de pré-processamento foi fornecida."
            )
        print("--- Executando pré-processamento dedicado ---")
        processed_df = preprocess_fn(config)
    else:
        print(
            f"--- Arquivo processado encontrado em {processed_path}. Pulando pré-processamento. ---"
        )

    if processed_df is None:
        processed_df = pd.read_csv(processed_path)

    if eda_plotter:
        try:
            print("--- Gerando gráficos exploratórios do dataset ---")
            eda_plotter(processed_df.copy(), config)
        except Exception as exc:
            print(f"AVISO: Falha ao gerar gráficos exploratórios: {exc}")

    models_to_run = list(config.get("models", {}).keys())
    predictions: Dict[str, pd.DataFrame] = {}

    for model_name in models_to_run:
        print(f"\n{'=' * 20} PROCESSANDO MODELO: {model_name.upper()} {'=' * 20}")
        run_command(
            [
                "poetry",
                "run",
                "python",
                "train.py",
                "--model",
                model_name,
                "--config",
                config_path,
            ]
        )
        run_command(
            [
                "poetry",
                "run",
                "python",
                "evaluate.py",
                "--model",
                model_name,
                "--config",
                config_path,
            ]
        )

        prediction_path = (
            Path(config["paths"]["results"]) / f"{model_name}_predictions.csv"
        )
        if prediction_path.exists():
            predictions[model_name] = pd.read_csv(prediction_path)
        else:
            print(
                f"AVISO: Arquivo de predição não encontrado para '{model_name}' em {prediction_path}"
            )

    if performance_plotter and predictions:
        print("\n--- Gerando gráficos de avaliação dos modelos ---")
        performance_plotter(predictions, config)
    else:
        print(
            "AVISO: Nenhum gráfico de avaliação foi gerado (sem predições ou plotter ausente)."
        )

    print("\n=== Pipeline concluído ===")
