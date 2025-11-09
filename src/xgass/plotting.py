# file: ./src/xgass/plotting.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_figure(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _plot_target_distribution(
    df: pd.DataFrame, target: str, out_dir: Path
) -> Optional[Path]:
    if target not in df.columns:
        print(f"AVISO: Coluna alvo '{target}' não encontrada. Pulando histograma.")
        return None

    clean = df[target].dropna()
    if clean.empty:
        print("AVISO: Sem dados válidos para o histograma da massa de HI.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    clean.hist(ax=ax, bins=30, color="#1f77b4", edgecolor="black")
    ax.set_xlabel(r"$\log M_{HI} [M_{\odot}]$")
    ax.set_ylabel("Número de galáxias")
    ax.set_title("Distribuição de massa de HI (xGASS)")
    return _save_figure(fig, out_dir, "hist_lgMHI")


def _plot_scatter(
    df: pd.DataFrame,
    x: str,
    target: str,
    out_dir: Path,
    *,
    color: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    suffix: str = "",
) -> Optional[Path]:
    cols = [x, target] + ([color] if color else [])
    if any(col not in df.columns for col in cols):
        print(f"AVISO: Colunas {cols} não encontradas. Pulando gráfico '{x}'.")
        return None

    clean = df[cols].dropna()
    if clean.empty:
        print(f"AVISO: Dados insuficientes para o gráfico '{x}'.")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        clean[x],
        clean[target],
        c=clean[color] if color else "#1f77b4",
        cmap="plasma" if color else None,
        s=25,
        alpha=0.8,
        edgecolor="none",
    )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or target)
    ax.set_title(title or f"{target} vs {x}")
    if color:
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color)
    safe_suffix = suffix or x
    return _save_figure(fig, out_dir, f"scatter_{target}_vs_{safe_suffix}")


def _plot_correlation_matrix(
    df: pd.DataFrame, cols: List[str], out_dir: Path
) -> Optional[Path]:
    available = [col for col in cols if col in df.columns]
    if len(available) < 2:
        print("AVISO: Colunas insuficientes para matriz de correlação.")
        return None

    corr = df[available].apply(pd.to_numeric, errors="coerce").corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right")
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(available)
    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(
                j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", color="black"
            )
    ax.set_title("Correlação de Pearson (variáveis selecionadas)")
    fig.colorbar(im, ax=ax, label="coeficiente")
    return _save_figure(fig, out_dir, "correlation_matrix")


def _plot_gas_fraction(df: pd.DataFrame, target: str, out_dir: Path) -> Optional[Path]:
    required = ["lgMstar", "NUVr", target]
    if any(col not in df.columns for col in required):
        print("AVISO: Colunas insuficientes para a fração de gás.")
        return None

    clean = df[required].dropna()
    if clean.empty:
        print("AVISO: Dados insuficientes para a fração de gás.")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(
        clean["lgMstar"],
        clean[target] - clean["lgMstar"],
        c=clean["NUVr"],
        cmap="viridis",
        s=25,
        alpha=0.85,
        edgecolor="none",
    )
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel(r"$\log M_* [M_{\odot}]$")
    ax.set_ylabel(r"$\log (M_{HI}/M_*)$")
    ax.set_title("Fração de gás atômico colorida por NUV-r")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("NUV - r")
    return _save_figure(fig, out_dir, "gas_fraction_vs_mass")


def generate_xgass_figures(df: pd.DataFrame, config: dict) -> List[Path]:
    """
    Cria os gráficos exploratórios utilizados no estudo do dataset xGASS.
    """
    target = config["target_column"]
    out_dir = Path(config["paths"]["plots"])

    figure_paths: List[Path] = []
    maybe_plot = _plot_target_distribution(df, target, out_dir)
    if maybe_plot:
        figure_paths.append(maybe_plot)

    figure_paths.extend(
        filter(
            None,
            [
                _plot_scatter(
                    df,
                    "lgMstar",
                    target,
                    out_dir,
                    color="NUVr",
                    xlabel=r"$\log M_* [M_{\odot}]$",
                    ylabel=r"$\log M_{HI} [M_{\odot}]$",
                    title="Massa estelar vs massa de HI",
                    suffix="lgMstar",
                ),
                _plot_scatter(
                    df,
                    "lgSFR_tot_median",
                    target,
                    out_dir,
                    xlabel=r"$\log \mathrm{SFR} [M_{\odot}/ano]$",
                    ylabel=r"$\log M_{HI} [M_{\odot}]$",
                    title="Reservatório de HI vs taxa de formação estelar",
                    suffix="lgSFR",
                ),
                _plot_scatter(
                    df,
                    "W50cor",
                    target,
                    out_dir,
                    xlabel="W50_cor [km/s]",
                    ylabel=r"$\log M_{HI} [M_{\odot}]$",
                    title="Largura da linha HI vs massa de HI",
                    suffix="W50cor",
                ),
                _plot_scatter(
                    df,
                    "INCL",
                    target,
                    out_dir,
                    xlabel="Inclinação [graus]",
                    ylabel=r"$\log M_{HI} [M_{\odot}]$",
                    title="Dependência com inclinação",
                    suffix="incl",
                ),
                _plot_gas_fraction(df, target, out_dir),
                _plot_correlation_matrix(
                    df,
                    [target]
                    + [
                        "lgMstar",
                        "NUVr",
                        "lgmust",
                        "lgSFR_tot_median",
                        "W50cor",
                        "INCL",
                        "lvir_ratB",
                    ],
                    out_dir,
                ),
            ],
        )
    )

    if figure_paths:
        print("--- Gráficos xGASS salvos ---")
        for path in figure_paths:
            print(f"- {path}")

    return figure_paths
