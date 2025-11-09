# file: ./src/xgass/preprocess.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


def _collect_required_columns(config: dict) -> List[str]:
    target = config["target_column"]
    ordered_features: List[str] = []
    for model_cfg in config.get("models", {}).values():
        for feature in model_cfg.get("features", []):
            if feature not in ordered_features and feature != target:
                ordered_features.append(feature)
    return [target] + ordered_features


def _load_fits_table(path: Path) -> pd.DataFrame:
    with fits.open(path) as hdul:
        table = Table(hdul[1].data)
        return table.to_pandas()


def _replace_sentinels(df: pd.DataFrame, sentinels: Iterable[float]) -> pd.DataFrame:
    if not sentinels:
        return df
    return df.replace(list(sentinels), np.nan)


def prepare_xgass_dataset(config: dict):
    """
    Carrega, limpa e seleciona as colunas necessárias do dataset xGASS.
    """
    paths = config["paths"]
    raw_path = Path(paths["raw_data"])
    processed_path = Path(paths["processed_data"])
    print(f"Lendo catálogo xGASS de '{raw_path}'")

    if not raw_path.exists():
        raise FileNotFoundError(f"Arquivo FITS não encontrado: {raw_path}")

    df = _load_fits_table(raw_path)
    sentinels = config.get("data_processing", {}).get("sentinel_values", [-99.0])
    df = _replace_sentinels(df, sentinels)

    required_columns = _collect_required_columns(config)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"As seguintes colunas necessárias não foram encontradas no catálogo xGASS: {missing}"
        )

    processed = df[required_columns].apply(pd.to_numeric, errors="coerce")
    before_drop = len(processed)
    processed = processed.dropna()

    os.makedirs(processed_path.parent, exist_ok=True)
    processed.to_csv(processed_path, index=False)

    print(
        f"Dataset xGASS processado: {before_drop} -> {len(processed)} linhas válidas."
        f" Arquivo salvo em: {processed_path}"
    )
    return processed
