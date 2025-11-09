# file: ./src/xgass/hyperparameter_search.py
from __future__ import annotations

import copy
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from src.utils.commands import run_command
from src.xgass.preprocess import prepare_xgass_dataset

COMMON_HP_SPACE = {
    "learning_rate": [5e-4, 7e-4, 1e-3],
    "batch_size": [128, 256],
    "dropout": [0.2, 0.3],
    "hidden_layers": [[256, 128, 64], [128, 64]],
    "core_layers": [[512, 256], [256, 128]],
    "head_layers": [[128, 64], [64, 32]],
}


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _ensure_processed_dataset(config: dict) -> None:
    processed_path = Path(config["paths"]["processed_data"])
    if not processed_path.exists():
        prepare_xgass_dataset(config)


def _build_search_combinations(model_cfg: dict) -> List[Dict[str, object]]:
    model_type = model_cfg["type"]
    keys: List[str] = ["learning_rate", "batch_size"]

    if model_type in {"vanilla", "bnn"}:
        keys.append("hidden_layers")
        if "dropout" in model_cfg:
            keys.append("dropout")
    elif model_type == "dbnn":
        keys.extend(["core_layers", "head_layers"])

    combinations: List[Dict[str, object]] = []
    options = [COMMON_HP_SPACE[key] for key in keys]
    for values in itertools.product(*options):
        combo = {}
        for key, value in zip(keys, values):
            combo[key] = value
        combinations.append(combo)
    return combinations


def _write_temp_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def _evaluate_trial(
    model_name: str,
    config_path: Path,
    trial_results_dir: Path,
    metric: str,
) -> Optional[float]:
    run_command(
        [
            "poetry",
            "run",
            "python",
            "train.py",
            "--model",
            model_name,
            "--config",
            str(config_path),
        ]
    )

    history_path = trial_results_dir / f"{model_name}_history.csv"
    if not history_path.exists():
        print(f"AVISO: histórico não encontrado em {history_path}")
        return None

    history = pd.read_csv(history_path)
    if metric not in history.columns:
        print(f"AVISO: métrica '{metric}' não encontrada no histórico {history_path}")
        return None

    return float(history[metric].min())


def optimize_xgass_hyperparameters(config_path: str) -> str:
    """
    Realiza busca em grade simples para cada modelo utilizando o mesmo conjunto
    de valores de hiperparâmetros definidos em COMMON_HP_SPACE.

    Returns
    -------
    str
        Caminho para o arquivo de configuração com os melhores hiperparâmetros.
    """
    base_config = _load_config(config_path)
    search_cfg = base_config.get("hyperparameter_search", {})
    if not search_cfg.get("enabled", False):
        return config_path

    metric = search_cfg.get("metric", "val_loss")
    max_trials = int(search_cfg.get("max_trials", 4))
    work_dir = Path(search_cfg.get("work_dir", "results/xgass/hp_search/"))
    output_config_path = Path(
        search_cfg.get("output_config", "results/xgass/config_optimized.yaml")
    )

    _ensure_processed_dataset(base_config)
    os.makedirs(work_dir, exist_ok=True)

    best_params_per_model: Dict[str, Dict[str, object]] = {}

    for model_name, model_cfg in base_config.get("models", {}).items():
        print(f"\n--- Busca de hiperparâmetros para {model_name.upper()} ---")
        combinations = _build_search_combinations(model_cfg)
        limited_combos = combinations[:max_trials]

        best_metric: Optional[float] = None
        best_combo: Optional[Dict[str, object]] = None

        for trial_idx, combo in enumerate(limited_combos):
            trial_id = f"{model_name}_trial_{trial_idx}"
            trial_config = copy.deepcopy(base_config)
            trial_config["models"][model_name].update(combo)

            trial_root = work_dir / trial_id
            for key in ("results", "models", "plots"):
                if key in trial_config["paths"]:
                    trial_path = trial_root / key
                    trial_config["paths"][key] = str(trial_path)

            temp_config_path = work_dir / f"{trial_id}.yaml"
            _write_temp_config(trial_config, temp_config_path)

            try:
                metric_value = _evaluate_trial(
                    model_name,
                    temp_config_path,
                    Path(trial_config["paths"]["results"]),
                    metric,
                )
            except RuntimeError as exc:
                print(f"AVISO: falha no trial {trial_id}: {exc}")
                continue

            if metric_value is None:
                continue

            print(f"Trial {trial_id}: {metric}={metric_value:.4f} com {combo}")

            if best_metric is None or metric_value < best_metric:
                best_metric = metric_value
                best_combo = combo

        if best_combo:
            best_params_per_model[model_name] = best_combo
            base_config["models"][model_name].update(best_combo)
            print(
                f"Melhor combinação para {model_name.upper()}: "
                f"{json.dumps(best_combo, ensure_ascii=False)} (metric={best_metric:.4f})"
            )
        else:
            print(f"Não foi possível determinar melhor combinação para {model_name}.")

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(base_config, stream, sort_keys=False)

    print(f"\nConfiguração otimizada salva em: {output_config_path}")
    return str(output_config_path)
