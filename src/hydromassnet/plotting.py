import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, r2_score

LINE_WIDTH = 2.1
TITLE_FONT_SIZE = 18
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 15
LEGEND_FONT_SIZE = 16
LEGEND_TEXT_COLOR = "#000000"
ANNOTATION_FONT_SIZE = 14
REGION_LABEL_SIZE = 16
COLORBAR_LABEL_SIZE = 16
TICK_LENGTH = 7
TICK_PADDING = 7

TRAIN_LOSS_COLOR = "#0a5fb4"
VAL_LOSS_COLOR = "#c0362c"
TRAIN_MAE_COLOR = "#1b8e3e"
VAL_MAE_COLOR = "#8b3fbf"
CONFIDENCE_68_COLOR = "#3d7dbf"
CONFIDENCE_95_COLOR = "#0f3e79"
CONFIDENCE_68_ALPHA = 0.35
CONFIDENCE_95_ALPHA = 0.18
CHECKPOINT_COLOR = "#8c510a"
CHECKPOINT_ALPHA = 0.6
CHECKPOINT_SIZE = 48
CHECKPOINT_LINEWIDTH = 1.05
OPTIMAL_MARKER_COLOR = "#4a1486"
DENSITY_CMAP = "magma"
LOW_DENSITY_SHADE_COLOR = "#fff7ec"
LOW_DENSITY_PERCENTILE = 25
ISOLATION_THRESHOLD = 3
ISOLATED_MARKER_STYLE = dict(
    marker="X", s=46, linewidth=1.05, facecolors="none", edgecolors="#7f2704"
)
TRUE_MHI_LABEL = r"True $\log_{10} M_{\mathrm{HI}}\; (M_\odot)$"
PRED_MHI_LABEL = r"Predicted $\log_{10} M_{\mathrm{HI}}\; (M_\odot)$"
LOSS_AXIS_LABEL = "Loss (log scale)"
MAE_AXIS_LABEL = "Mean absolute error (log scale)"
EPOCH_AXIS_LABEL = "Epoch"
HEXBIN_CBAR_LABEL = r"$\log_{10}$ point density"
DENSITY_BAR_LABEL = r"$\log_{10}$ galaxy density"
FEATURE_DENSITY_LABEL = "Probability density"
CHECKPOINT_LABEL = "Validation checkpoints"
BEST_CHECKPOINT_TEXT = "Best checkpoint (validation)"
SIGMA_ERROR_LABEL = r"$\pm 1\sigma$ error bars"
FEATURE_LABEL_MAP = {
    "iMAG": "$i$ magnitude (mag)",
    "e_iMAG": "$i$ magnitude uncertainty (mag)",
    "logMsT": r"$\log_{10} M_*^{\mathrm{tot}}\,(M_\odot)$",
    "e_logMsT": r"$\sigma(\log_{10} M_*^{\mathrm{tot}})$",
    "logMsM": r"$\log_{10} M_*^{\mathrm{morph}}\,(M_\odot)$",
    "logMsG": r"$\log_{10} M_*^{\mathrm{group}}\,(M_\odot)$",
    "Ag": "$A_g$ extinction (mag)",
    "Ai": "$A_i$ extinction (mag)",
    "Dist": "Distance (Mpc)",
    "RVel": "Radial velocity (km s$^{-1}$)",
    "logSFR22": r"$\log_{10} \mathrm{SFR}_{22}\,(M_\odot\,\mathrm{yr}^{-1})$",
    "logSFRN": r"$\log_{10} \mathrm{SFR}_{NUV}$",
    "logSFRG": r"$\log_{10} \mathrm{SFR}_{g}$",
    "surface_brightness_proxy": "Surface-brightness proxy",
    "g-i": "$g-i$ colour (mag)",
    "b/a": "Axis ratio $b/a$",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": False,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": AXIS_LABEL_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "axes.formatter.use_mathtext": True,
        "text.color": LEGEND_TEXT_COLOR,
        "axes.labelcolor": LEGEND_TEXT_COLOR,
        "axes.edgecolor": LEGEND_TEXT_COLOR,
        "xtick.color": LEGEND_TEXT_COLOR,
        "ytick.color": LEGEND_TEXT_COLOR,
        "axes.titlepad": 12,
        "axes.labelpad": 8,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "xtick.major.pad": TICK_PADDING,
        "ytick.major.pad": TICK_PADDING,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

sns.set_theme(style="whitegrid", rc=plt.rcParams)


def _clip_positive(values, floor=1e-6):
    arr = np.asarray(values)
    return np.clip(arr, floor, None)


def _calculate_checkpoint_epochs(val_loss_series):
    if val_loss_series is None or len(val_loss_series) == 0:
        return []

    best_so_far = np.inf
    checkpoint_epochs = []
    for idx, value in enumerate(val_loss_series):
        if value < best_so_far - 1e-9:
            checkpoint_epochs.append(idx + 1)
            best_so_far = value
    return checkpoint_epochs


def _annotate_optimal_point(ax, epoch, value, text):
    if epoch is None or value is None:
        return
    ax.scatter(
        epoch,
        value,
        marker="*",
        s=170,
        color=OPTIMAL_MARKER_COLOR,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    def _rel(val, limits):
        span = limits[1] - limits[0]
        if span == 0:
            return 0.5
        return (val - limits[0]) / span

    x_rel = _rel(epoch, xlim)
    y_rel = _rel(value, ylim)

    x_offset = 12
    ha = "left"
    if x_rel > 0.8:
        x_offset = -12
        ha = "right"
    elif x_rel < 0.2:
        x_offset = 12
        ha = "left"

    y_offset = 16
    va = "bottom"
    if y_rel > 0.8:
        y_offset = -16
        va = "top"

    ax.annotate(
        text,
        xy=(epoch, value),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=ANNOTATION_FONT_SIZE,
        arrowprops=dict(
            arrowstyle="->",
            color=OPTIMAL_MARKER_COLOR,
            linewidth=1.2,
            mutation_scale=12,
            shrinkB=4,
        ),
    )


def _identify_isolated_points(
    x, y, xedges, yedges, raw_hist, threshold=ISOLATION_THRESHOLD
):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0:
        return np.array([], dtype=bool)

    x_idx = np.digitize(x, xedges) - 1
    y_idx = np.digitize(y, yedges) - 1
    valid = (
        (x_idx >= 0)
        & (x_idx < len(xedges) - 1)
        & (y_idx >= 0)
        & (y_idx < len(yedges) - 1)
    )
    mask = np.zeros_like(x, dtype=bool)
    mask[valid] = raw_hist[x_idx[valid], y_idx[valid]] <= threshold
    return mask


def _format_feature_label(name):
    if not name:
        return ""
    if name in FEATURE_LABEL_MAP:
        return FEATURE_LABEL_MAP[name]
    label = name.replace("_", " ").strip()
    if not label:
        return name
    # Preserve lowercase "log" prefixes
    if label.lower().startswith("log") and len(label) > 3:
        label = r"$\log$ " + label[3:].strip()
    return label.title()


def _styled_legend(ax, handles=None, labels=None, **kwargs):
    params = {
        "frameon": False,
        "fontsize": LEGEND_FONT_SIZE,
        "labelcolor": LEGEND_TEXT_COLOR,
        "handletextpad": 0.8,
        "borderpad": 0.4,
        "loc": kwargs.pop("loc", "upper right"),
        "ncol": kwargs.pop("ncol", 1),
        "borderaxespad": 0.6,
        "columnspacing": 1.0,
        "handlelength": 1.8,
    }
    params.update(kwargs)

    if handles is not None and labels is not None:
        legend = ax.legend(handles, labels, **params)
    elif handles is not None:
        legend = ax.legend(handles=handles, **params)
    else:
        legend = ax.legend(**params)

    if legend and legend.get_frame() is not None:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("none")
    return legend


def plot_predictions_overview(predictions, config):
    """Generate a consolidated 'true vs predicted values' grid for every model."""
    plots_dir = config["paths"]["plots"]

    if not predictions:
        print("No predictions available for overview plot.")
        return

    # Get global min/max values and density range for consistent scaling
    min_val, max_val = np.inf, -np.inf
    max_density = 0
    for _, df in predictions.items():
        min_val = min(min_val, df["y_true"].min(), df["y_pred_mean"].min())
        max_val = max(max_val, df["y_true"].max(), df["y_pred_mean"].max())

        hist, _, _ = np.histogram2d(df["y_true"], df["y_pred_mean"], bins=50)
        max_density = max(max_density, hist.max())

    padding = (max_val - min_val) * 0.05
    min_val -= padding
    max_val += padding

    n_models = len(predictions)
    cols = 2
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.8 * cols, 5.8 * rows))
    axes = np.atleast_1d(axes).flatten()

    color_norm = mcolors.LogNorm(vmin=1, vmax=max(1, max_density))
    color_reference = None

    for ax, (model_name, df) in zip(axes, predictions.items()):
        y_true, y_pred = df["y_true"], df["y_pred_mean"]
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        hb = ax.hexbin(
            y_true,
            y_pred,
            gridsize=45,
            cmap=DENSITY_CMAP,
            mincnt=1,
            linewidths=0,
            norm=color_norm,
        )
        color_reference = hb
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="crimson",
            linewidth=LINE_WIDTH,
            label="Identity line (y = x)",
        )

        ax.set_title(
            f"{model_name.upper()} - $R^2={r2:.3f}$ | RMSE={rmse:.3f}",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.set_xlabel(TRUE_MHI_LABEL, fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(PRED_MHI_LABEL, fontsize=AXIS_LABEL_SIZE)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, alpha=0.32)
        ax.tick_params(length=TICK_LENGTH, pad=TICK_PADDING)
        ax.margins(x=0.02, y=0.02)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            _styled_legend(ax, loc="upper left")

    # Hide unused axes if the grid is not full
    for ax in axes[n_models:]:
        ax.set_visible(False)

    fig.suptitle(
        r"True vs. predicted $\log_{10} M_{\mathrm{HI}}$ (shared density scale)",
        fontsize=TITLE_FONT_SIZE + 1,
        fontweight="bold",
        y=0.99,
    )

    if color_reference is not None:
        cbar_ax = fig.add_axes([0.94, 0.12, 0.015, 0.74])
        cbar = fig.colorbar(color_reference, cax=cbar_ax)
        cbar.set_label(HEXBIN_CBAR_LABEL, fontsize=COLORBAR_LABEL_SIZE, labelpad=10)
        cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, length=TICK_LENGTH)

    fig.subplots_adjust(
        left=0.08, right=0.92, bottom=0.08, top=0.92, wspace=0.25, hspace=0.3
    )
    plot_path = os.path.join(plots_dir, "predictions_overview.png")
    plt.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Prediction overview saved: {plot_path}")


def plot_confidence_intervals(predictions, config):
    """Plot individual predictions with confidence intervals for Bayesian models."""
    bayesian_models = {k: v for k, v in predictions.items() if k in ["bnn", "dbnn"]}
    if not bayesian_models:
        print("No Bayesian models found for confidence interval plots.")
        return

    plots_dir = config["paths"]["plots"]

    for model_name, df in bayesian_models.items():
        fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.6))

        df_sample = df.sample(n=min(len(df), 600), random_state=config["seed"])
        df_sorted = df_sample.sort_values(by="y_true").reset_index()

        y_true = df_sorted["y_true"]
        y_mean = df_sorted["y_pred_mean"]
        y_std = _clip_positive(df_sorted["y_pred_std"])
        ci68 = y_std
        ci95 = 1.96 * y_std

        ax.fill_between(
            y_true,
            y_mean - ci95,
            y_mean + ci95,
            color=CONFIDENCE_95_COLOR,
            alpha=CONFIDENCE_95_ALPHA,
            label="95% credible band",
            zorder=1,
        )
        ax.fill_between(
            y_true,
            y_mean - ci68,
            y_mean + ci68,
            color=CONFIDENCE_68_COLOR,
            alpha=CONFIDENCE_68_ALPHA,
            label="68% credible band",
            zorder=2,
        )

        ax.errorbar(
            y_true,
            y_mean,
            yerr=ci68,
            fmt="none",
            ecolor=CONFIDENCE_68_COLOR,
            elinewidth=1.15,
            alpha=0.45,
            capsize=2.4,
            label=SIGMA_ERROR_LABEL,
            zorder=3,
        )
        ax.scatter(
            y_true,
            y_mean,
            color="#1a1a1a",
            s=18,
            alpha=0.72,
            label="Mean prediction",
            zorder=4,
        )
        ax.plot(
            y_true,
            y_true,
            linestyle="--",
            linewidth=LINE_WIDTH,
            color="crimson",
            label="Identity line (y = x)",
            zorder=5,
        )

        legend_handles = [
            Patch(
                facecolor=CONFIDENCE_95_COLOR,
                alpha=CONFIDENCE_95_ALPHA,
                label="95% credible band",
            ),
            Patch(
                facecolor=CONFIDENCE_68_COLOR,
                alpha=CONFIDENCE_68_ALPHA,
                label="68% credible band",
            ),
            Line2D(
                [0],
                [0],
                color=CONFIDENCE_68_COLOR,
                linewidth=1.6,
                label=SIGMA_ERROR_LABEL,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="#1a1a1a",
                label="Mean prediction",
            ),
            Line2D(
                [0],
                [0],
                linestyle="--",
                linewidth=LINE_WIDTH,
                color="crimson",
                label="Identity line (y = x)",
            ),
        ]

        ax.set_title(
            f"{model_name.upper()} - Uncertainty calibration",
            fontsize=TITLE_FONT_SIZE,
            fontweight="bold",
        )
        ax.set_xlabel(TRUE_MHI_LABEL, fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(PRED_MHI_LABEL, fontsize=AXIS_LABEL_SIZE)
        ax.grid(True, alpha=0.32)
        ax.tick_params(length=TICK_LENGTH, pad=TICK_PADDING)
        ax.margins(x=0.02, y=0.02)
        _styled_legend(ax, handles=legend_handles, loc="upper left", ncol=2)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f"{model_name}_confidence_intervals.png")
        plt.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Confidence interval plot saved: {plot_path}")


def plot_training_metrics(config):
    """Plot individual training metrics (loss and MAE) for each model."""
    results_dir = config["paths"]["results"]
    plots_dir = config["paths"]["plots"]

    # Find all history files
    history_files = glob.glob(os.path.join(results_dir, "*_history.csv"))

    if not history_files:
        print("No training history files found. Skipping training metrics plots.")
        return

    for file_path in history_files:
        model_name = os.path.basename(file_path).replace("_history.csv", "").upper()
        try:
            df = pd.read_csv(file_path)
            epochs = np.arange(1, len(df) + 1)

            val_loss_series = None
            if "val_loss" in df.columns:
                val_loss_series = np.nan_to_num(df["val_loss"].to_numpy(), nan=np.inf)
            checkpoint_epochs = _calculate_checkpoint_epochs(val_loss_series)
            best_epoch = None
            best_val = None
            if val_loss_series is not None and np.isfinite(val_loss_series).any():
                best_idx = int(np.nanargmin(val_loss_series))
                best_epoch = best_idx + 1
                best_val = df["val_loss"].iloc[best_idx]

            # Create separate loss plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(
                epochs,
                _clip_positive(df["loss"]),
                label="Training loss",
                linewidth=LINE_WIDTH,
                color=TRAIN_LOSS_COLOR,
            )

            if "val_loss" in df.columns:
                ax.plot(
                    epochs,
                    _clip_positive(df["val_loss"]),
                    label="Validation loss",
                    linewidth=LINE_WIDTH,
                    linestyle="--",
                    color=VAL_LOSS_COLOR,
                )
                if checkpoint_epochs:
                    checkpoint_vals = _clip_positive(
                        df["val_loss"].iloc[np.array(checkpoint_epochs) - 1]
                    )
                    ax.scatter(
                        checkpoint_epochs,
                        checkpoint_vals,
                        marker="o",
                        s=CHECKPOINT_SIZE,
                        facecolors="none",
                        edgecolors=CHECKPOINT_COLOR,
                        linewidths=CHECKPOINT_LINEWIDTH,
                        alpha=CHECKPOINT_ALPHA,
                        label=CHECKPOINT_LABEL,
                        zorder=4,
                    )

            _annotate_optimal_point(ax, best_epoch, best_val, BEST_CHECKPOINT_TEXT)

            ax.set_xlabel(EPOCH_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE)
            loss_label = "Loss" if model_name.upper() == "BASELINE" else LOSS_AXIS_LABEL
            ax.set_ylabel(loss_label, fontsize=AXIS_LABEL_SIZE)
            ax.set_title(
                f"{model_name.upper()} - Loss curve",
                fontsize=TITLE_FONT_SIZE,
                fontweight="bold",
            )
            ax.set_yscale("log")
            ax.grid(True, alpha=0.32)
            ax.tick_params(length=TICK_LENGTH, pad=TICK_PADDING)
            ax.margins(x=0.02, y=0.02)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                _styled_legend(ax, handles=handles, labels=labels, loc="upper right")

            plt.tight_layout()
            loss_plot_path = os.path.join(plots_dir, f"{model_name.lower()}_loss.png")
            plt.savefig(loss_plot_path, format="png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Loss plot saved: {loss_plot_path}")

            # Create separate MAE plot
            if "mae" in df.columns or "val_mae" in df.columns:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                if "mae" in df.columns:
                    ax.plot(
                        epochs,
                        _clip_positive(df["mae"]),
                        label="Training MAE",
                        linewidth=LINE_WIDTH,
                        color=TRAIN_MAE_COLOR,
                    )
                if "val_mae" in df.columns:
                    ax.plot(
                        epochs,
                        _clip_positive(df["val_mae"]),
                        label="Validation MAE",
                        linewidth=LINE_WIDTH,
                        linestyle="--",
                        color=VAL_MAE_COLOR,
                    )
                    if checkpoint_epochs:
                        checkpoint_mae = _clip_positive(
                            df["val_mae"].iloc[np.array(checkpoint_epochs) - 1]
                        )
                        ax.scatter(
                            checkpoint_epochs,
                            checkpoint_mae,
                            marker="o",
                            s=CHECKPOINT_SIZE - 6,
                            facecolors="none",
                            edgecolors=CHECKPOINT_COLOR,
                            linewidths=CHECKPOINT_LINEWIDTH,
                            alpha=CHECKPOINT_ALPHA,
                            label=CHECKPOINT_LABEL,
                            zorder=4,
                        )

                if best_epoch is not None and "val_mae" in df.columns:
                    best_mae = df["val_mae"].iloc[best_epoch - 1]
                    _annotate_optimal_point(
                        ax, best_epoch, best_mae, BEST_CHECKPOINT_TEXT
                    )

                ax.set_xlabel(EPOCH_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE)
                ax.set_ylabel(MAE_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE)
                ax.set_title(
                    f"{model_name.upper()} - MAE curve",
                    fontsize=TITLE_FONT_SIZE,
                    fontweight="bold",
                )
                ax.set_yscale("log")
                ax.grid(True, alpha=0.32)
                ax.tick_params(length=TICK_LENGTH, pad=TICK_PADDING)
                ax.margins(x=0.02, y=0.02)

                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    _styled_legend(
                        ax, handles=handles, labels=labels, loc="upper right"
                    )

                plt.tight_layout()
                mae_plot_path = os.path.join(plots_dir, f"{model_name.lower()}_mae.png")
                plt.savefig(mae_plot_path, format="png", dpi=300, bbox_inches="tight")
                plt.close()
                print(f"MAE plot saved: {mae_plot_path}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue


def plot_correlation_matrix(config):
    """Plot Pearson correlation matrix for the features used in models."""
    from src.hydromassnet.data import DataHandler

    plots_dir = config["paths"]["plots"]

    try:
        # Get all unique features from all models
        all_features = set()
        for model_name, model_config in config["models"].items():
            if "features" in model_config:
                all_features.update(model_config["features"])

        if not all_features:
            print("No features found in model configurations.")
            return

        all_features = sorted(list(all_features))

        # Load data using the most comprehensive feature set
        data_handler = DataHandler(config, feature_override=all_features)
        x_train, y_train, x_val, y_val, x_test, y_test, features = (
            data_handler.get_full_dataset_and_splits()
        )

        # Combine all data for correlation analysis
        import pandas as pd

        all_data = pd.concat(
            [
                pd.DataFrame(x_train, columns=features),
                pd.DataFrame(x_val, columns=features),
                pd.DataFrame(x_test, columns=features),
            ],
            ignore_index=True,
        )

        # Calculate Pearson correlation matrix
        correlation_matrix = all_data.corr(method="pearson")

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(11, 9))

        # Create a mask for the upper triangle (optional, for cleaner look)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Generate heatmap
        heatmap = sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"label": r"Pearson correlation coefficient ($r$)"},
            annot_kws={"size": 12, "color": "#1a1a1a"},
            linewidths=0.4,
            linecolor="white",
        )

        ax.set_title(
            "Pearson correlation heatmap (masked)",
            fontsize=TITLE_FONT_SIZE - 2,
            fontweight="bold",
            pad=20,
        )

        # Rotate labels for better readability
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=40,
            ha="right",
            fontsize=TICK_LABEL_SIZE,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=TICK_LABEL_SIZE,
        )
        ax.tick_params(length=TICK_LENGTH, pad=TICK_PADDING)

        # Style colorbar for legibility
        if heatmap.collections:
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, length=TICK_LENGTH)
            cbar.set_label(
                r"Pearson correlation coefficient ($r$)",
                fontsize=COLORBAR_LABEL_SIZE,
                labelpad=10,
            )

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "correlation_matrix.png")
        plt.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Correlation matrix plot saved: {plot_path}")

    except Exception as e:
        print(f"Error generating correlation matrix: {e}")


def plot_correlation_matrix_complete(config):
    """Plot lower-triangular Pearson correlation matrix (no redundant cells)."""
    from src.hydromassnet.data import DataHandler

    plots_dir = config["paths"]["plots"]

    try:
        # Get all unique features from all models
        all_features = set()
        for model_name, model_config in config["models"].items():
            if "features" in model_config:
                all_features.update(model_config["features"])

        if not all_features:
            print("No features found in model configurations.")
            return

        all_features = sorted(list(all_features))

        # Load data using the most comprehensive feature set
        data_handler = DataHandler(config, feature_override=all_features)
        x_train, y_train, x_val, y_val, x_test, y_test, features = (
            data_handler.get_full_dataset_and_splits()
        )

        # Combine all data for correlation analysis
        import pandas as pd

        all_data = pd.concat(
            [
                pd.DataFrame(x_train, columns=features),
                pd.DataFrame(x_val, columns=features),
                pd.DataFrame(x_test, columns=features),
            ],
            ignore_index=True,
        )

        # Calculate Pearson correlation matrix
        correlation_matrix = all_data.corr(method="pearson")

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(11, 9))

        # Mask upper triangle to show only non-repeated correlations
        lower_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)

        heatmap = sns.heatmap(
            correlation_matrix,
            mask=lower_mask,
            annot=True,
            cmap="RdBu_r",
            center=0,
            square=True,
            fmt=".2f",
            cbar_kws={"label": r"Pearson correlation coefficient ($r$)"},
            annot_kws={"size": 12, "color": "#1a1a1a"},
            linewidths=0.4,
            linecolor="white",
        )

        ax.set_title(
            "Pearson correlation matrix (lower triangle)",
            fontsize=TITLE_FONT_SIZE - 2,
            fontweight="bold",
            pad=20,
        )

        # Rotate labels for better readability
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=40,
            ha="right",
            fontsize=TICK_LABEL_SIZE,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
            fontsize=TICK_LABEL_SIZE,
        )
        ax.tick_params(length=TICK_LENGTH, pad=TICK_PADDING)

        if heatmap.collections:
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, length=TICK_LENGTH)
            cbar.set_label(
                r"Pearson correlation coefficient ($r$)",
                fontsize=COLORBAR_LABEL_SIZE,
                labelpad=10,
            )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "correlation_matrix_complete.png")
        plt.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Complete correlation matrix plot saved: {plot_path}")

    except Exception as e:
        print(f"Error generating complete correlation matrix: {e}")


def plot_feature_distributions(config):
    """Plot individual feature distributions to understand data characteristics."""
    from src.hydromassnet.data import DataHandler

    plots_dir = config["paths"]["plots"]

    try:
        # Get all unique features from all models
        all_features = set()
        for model_name, model_config in config["models"].items():
            if "features" in model_config:
                all_features.update(model_config["features"])

        if not all_features:
            print("No features found in model configurations.")
            return

        all_features = sorted(list(all_features))

        # Load data
        data_handler = DataHandler(config, feature_override=all_features)
        x_train, y_train, x_val, y_val, x_test, y_test, features = (
            data_handler.get_full_dataset_and_splits()
        )

        # Combine all data
        import pandas as pd

        all_data = pd.concat(
            [
                pd.DataFrame(x_train, columns=features),
                pd.DataFrame(x_val, columns=features),
                pd.DataFrame(x_test, columns=features),
            ],
            ignore_index=True,
        )

        # Create individual distribution plots
        n_features = len(features)
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_features == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i, feature in enumerate(features):
            row = i // cols
            col = i % cols
            if rows == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]

            # Plot histogram with KDE
            ax.hist(
                all_data[feature],
                bins=30,
                alpha=0.7,
                density=True,
                color="#4c8ccf",
                edgecolor="#0f1f2e",
            )

            # Add KDE curve
            from scipy import stats

            kde = stats.gaussian_kde(all_data[feature].dropna())
            x_range = np.linspace(all_data[feature].min(), all_data[feature].max(), 100)
            ax.plot(
                x_range,
                kde(x_range),
                "-",
                linewidth=2.2,
                color="#c0362c",
                label="Smoothed KDE",
            )

            feature_label = _format_feature_label(feature)
            ax.set_xlabel(feature_label, fontsize=AXIS_LABEL_SIZE)
            ax.set_ylabel(FEATURE_DENSITY_LABEL, fontsize=AXIS_LABEL_SIZE)
            ax.set_title(
                f"Distribution of {feature_label}", fontsize=TITLE_FONT_SIZE - 1
            )
            ax.tick_params(length=TICK_LENGTH, pad=TICK_PADDING)
            _styled_legend(ax, loc="upper right")
            ax.grid(True, alpha=0.32)

        # Hide empty subplots
        for i in range(len(features), rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, "feature_distributions.png")
        plt.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Feature distributions plot saved: {plot_path}")

    except Exception as e:
        print(f"Error generating feature distributions: {e}")


def plot_learning_curves(config):
    """Plot individual comprehensive learning curves for each model."""
    results_dir = config["paths"]["results"]
    plots_dir = config["paths"]["plots"]

    # Find all history files
    history_files = glob.glob(os.path.join(results_dir, "*_history.csv"))

    if not history_files:
        print("No training history files found. Skipping learning curves plots.")
        return

    for file_path in history_files:
        model_name = os.path.basename(file_path).replace("_history.csv", "").upper()
        try:
            df = pd.read_csv(file_path)
            epochs = np.arange(1, len(df) + 1)

            val_loss_series = None
            if "val_loss" in df.columns:
                val_loss_series = np.nan_to_num(df["val_loss"].to_numpy(), nan=np.inf)
            checkpoint_epochs = _calculate_checkpoint_epochs(val_loss_series)
            best_epoch = None
            best_val_loss = None
            if val_loss_series is not None and np.isfinite(val_loss_series).any():
                best_idx = int(np.nanargmin(val_loss_series))
                best_epoch = best_idx + 1
                best_val_loss = df["val_loss"].iloc[best_idx]

            fig, ax1 = plt.subplots(1, 1, figsize=(12.2, 6.4))
            ax2 = ax1.twinx()

            legend_handles = []

            loss_train_line = ax1.plot(
                epochs,
                _clip_positive(df["loss"]),
                color=TRAIN_LOSS_COLOR,
                linewidth=LINE_WIDTH,
                label="Training loss",
            )
            legend_handles.extend(loss_train_line)

            if "val_loss" in df.columns:
                val_loss_line = ax1.plot(
                    epochs,
                    _clip_positive(df["val_loss"]),
                    color=VAL_LOSS_COLOR,
                    linewidth=LINE_WIDTH,
                    linestyle="--",
                    label="Validation loss",
                )
                legend_handles.extend(val_loss_line)
                if checkpoint_epochs:
                    checkpoint_vals = _clip_positive(
                        df["val_loss"].iloc[np.array(checkpoint_epochs) - 1]
                    )
                    ax1.scatter(
                        checkpoint_epochs,
                        checkpoint_vals,
                        marker="o",
                        s=CHECKPOINT_SIZE + 6,
                        facecolors="none",
                        edgecolors=CHECKPOINT_COLOR,
                        linewidths=CHECKPOINT_LINEWIDTH,
                        alpha=CHECKPOINT_ALPHA,
                        zorder=4,
                    )

            _annotate_optimal_point(
                ax1, best_epoch, best_val_loss, BEST_CHECKPOINT_TEXT
            )

            if "mae" in df.columns:
                mae_train_line = ax2.plot(
                    epochs,
                    _clip_positive(df["mae"]),
                    color=TRAIN_MAE_COLOR,
                    linewidth=LINE_WIDTH,
                    label="Training MAE",
                )
                legend_handles.extend(mae_train_line)
            if "val_mae" in df.columns:
                mae_val_line = ax2.plot(
                    epochs,
                    _clip_positive(df["val_mae"]),
                    color=VAL_MAE_COLOR,
                    linewidth=LINE_WIDTH,
                    linestyle="--",
                    label="Validation MAE",
                )
                legend_handles.extend(mae_val_line)
                if checkpoint_epochs:
                    checkpoint_mae = _clip_positive(
                        df["val_mae"].iloc[np.array(checkpoint_epochs) - 1]
                    )
                    ax2.scatter(
                        checkpoint_epochs,
                        checkpoint_mae,
                        marker="o",
                        s=CHECKPOINT_SIZE,
                        facecolors="none",
                        edgecolors=CHECKPOINT_COLOR,
                        linewidths=CHECKPOINT_LINEWIDTH,
                        alpha=CHECKPOINT_ALPHA,
                        zorder=4,
                    )

            checkpoint_handle = None
            if checkpoint_epochs and (
                "val_loss" in df.columns or "val_mae" in df.columns
            ):
                checkpoint_handle = Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="none",
                    markeredgecolor=CHECKPOINT_COLOR,
                    label=CHECKPOINT_LABEL,
                )

            ax1.set_xlabel(EPOCH_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE)
            loss_label = "Loss" if model_name.upper() == "BASELINE" else LOSS_AXIS_LABEL
            ax1.set_ylabel(loss_label, fontsize=AXIS_LABEL_SIZE)
            ax1.tick_params(
                axis="both",
                labelcolor=LEGEND_TEXT_COLOR,
                length=TICK_LENGTH,
                pad=TICK_PADDING,
                labelsize=TICK_LABEL_SIZE,
                right=False,
                labelright=False,
            )
            ax1.set_yscale("log")
            ax1.grid(True, alpha=0.32)
            ax1.margins(x=0.02, y=0.02)
            ax1.spines["right"].set_visible(False)

            ax2.set_xlabel("")
            ax2.tick_params(axis="x", labelbottom=False, bottom=False)
            if model_name.upper() == "BASELINE":
                ax2.set_ylabel("")
                ax2.tick_params(labelleft=False, left=False)
            else:
                ax2.set_ylabel("")
            ax2.tick_params(
                axis="y",
                labelcolor=LEGEND_TEXT_COLOR,
                length=TICK_LENGTH,
                pad=TICK_PADDING,
                labelsize=TICK_LABEL_SIZE,
                labelright=False,
                right=False,
            )
            ax2.spines["right"].set_visible(False)
            ax2.set_yscale("log")

            legend_labels = [line.get_label() for line in legend_handles]
            if checkpoint_handle is not None:
                legend_handles.append(checkpoint_handle)
                legend_labels.append(checkpoint_handle.get_label())
            _styled_legend(
                ax1,
                handles=legend_handles,
                labels=legend_labels,
                loc="upper right",
                ncol=1,
            )

            ax1.set_title(
                f"{model_name.upper()} - Loss & MAE curves",
                fontsize=TITLE_FONT_SIZE,
                fontweight="bold",
                pad=24,
            )

            fig.tight_layout(rect=[0, 0, 1, 0.85])
            learning_plot_path = os.path.join(
                plots_dir, f"{model_name.lower()}_learning_curve.png"
            )
            plt.savefig(learning_plot_path, format="png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Learning curve plot saved: {learning_plot_path}")

        except Exception as e:
            print(f"Error plotting learning curve for {file_path}: {e}")
            continue


def plot_all(predictions, config):
    """Main function to call all plotting routines."""
    plots_dir = config["paths"]["plots"]
    os.makedirs(plots_dir, exist_ok=True)

    print("--- Generating correlation matrix plot (masked) ---")
    plot_correlation_matrix(config)

    print("--- Generating complete correlation matrix plot ---")
    plot_correlation_matrix_complete(config)

    print("--- Generating feature distribution plots ---")
    plot_feature_distributions(config)

    print("--- Generating individual prediction plots ---")
    plot_predictions_overview(predictions, config)

    print("--- Generating individual confidence interval plots ---")
    plot_confidence_intervals(predictions, config)

    print("--- Generating individual training metrics plots ---")
    plot_training_metrics(config)

    print("--- Generating individual learning curves ---")
    plot_learning_curves(config)


def plot_color_stellar_mass_diagram(config):
    """Generate color-stellar mass diagram with contour density plots split by morphology."""
    plots_dir = config["paths"]["plots"]

    try:
        # Load data
        df = pd.read_csv("data/hydromassnet_full_dataset_all_columns.csv")
        df = df.dropna(subset=["logMsT", "g-i", "b/a", "logSFR22"])

        # Filter reasonable ranges
        df = df[(df["logMsT"] > 8.5) & (df["logMsT"] < 12.0)]
        df = df[(df["g-i"] > 0.0) & (df["g-i"] < 3.5)]

        # Create morphology classification
        ba_median = df["b/a"].median()
        sfr_median = df["logSFR22"].median()

        df["morphology"] = "Late-type"
        early_type_mask = (df["b/a"] > ba_median) & (df["logSFR22"] < sfr_median)
        df.loc[early_type_mask, "morphology"] = "Early-type"

        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1, 0.8],
            width_ratios=[1, 1],
            hspace=0.25,
            wspace=0.22,
            left=0.07,
            right=0.9,
            bottom=0.08,
            top=0.92,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[1, :], sharex=ax1)
        cbar_ax = fig.add_axes([0.915, 0.23, 0.015, 0.6])

        def create_density_contour_plot(
            x, y, ax, title, panel_label, levels=12, add_legend_labels=False
        ):
            """Create log-scaled density contour plot for given data."""
            if len(x) < 10:
                ax.scatter(x, y, alpha=0.6, s=25, color="#636363")
                ax.set_title(
                    f"{panel_label}: {title}",
                    fontsize=TITLE_FONT_SIZE - 1,
                    fontweight="bold",
                )
                return None

            H_raw, xedges, yedges = np.histogram2d(x, y, bins=55)
            H_smooth = gaussian_filter(H_raw, sigma=1.1) + 1e-6
            H_plot = H_smooth.T
            X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

            positive_vals = H_plot[H_plot > 0]
            if positive_vals.size == 0:
                positive_vals = np.array([1])
            vmin = positive_vals.min()
            vmax = positive_vals.max()
            level_values = np.geomspace(vmin, vmax, levels)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

            contourf = ax.contourf(
                X,
                Y,
                H_plot,
                levels=level_values,
                cmap=DENSITY_CMAP,
                norm=norm,
                alpha=0.9,
                extend="max",
            )
            ax.contour(
                X,
                Y,
                H_plot,
                levels=level_values,
                colors="#3f007d",
                linewidths=0.65,
                alpha=0.8,
            )

            if positive_vals.size:
                low_threshold = np.percentile(positive_vals, LOW_DENSITY_PERCENTILE)
                low_mask = np.where(H_plot <= low_threshold, 1, 0)
                if np.any(low_mask):
                    ax.contourf(
                        X,
                        Y,
                        low_mask,
                        levels=[0.5, 1.5],
                        colors=[LOW_DENSITY_SHADE_COLOR],
                        alpha=0.25,
                        zorder=1,
                    )

            isolated_mask = _identify_isolated_points(x, y, xedges, yedges, H_raw)
            if isolated_mask.any():
                label = (
                    "Isolated galaxies (<= {} per bin)".format(ISOLATION_THRESHOLD)
                    if add_legend_labels
                    else None
                )
                ax.scatter(
                    x[isolated_mask],
                    y[isolated_mask],
                    label=label,
                    zorder=5,
                    **ISOLATED_MARKER_STYLE,
                )

            ax.set_title(
                f"{panel_label}: {title}",
                fontsize=TITLE_FONT_SIZE - 1,
                fontweight="bold",
                pad=8,
            )
            return contourf

        def add_region_labels(ax):
            """Annotate canonical colour–mass regions."""
            text_kwargs = dict(
                fontsize=REGION_LABEL_SIZE,
                color=LEGEND_TEXT_COLOR,
                ha="center",
                fontweight="semibold",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.65,
                    boxstyle="round,pad=0.25",
                ),
            )
            ax.text(10.8, 2.7, "red sequence", **text_kwargs)
            ax.text(10.2, 2.0, "green valley", **text_kwargs)
            ax.text(9.9, 1.5, "blue cloud", **text_kwargs)

        def add_separating_lines(ax, label=None):
            """Add approximate separating lines between regions."""
            x_range = np.linspace(8.5, 12.0, 100)
            red_green = 0.8 + 0.15 * (x_range - 10.0)
            green_blue = 0.6 + 0.12 * (x_range - 10.0)
            ax.plot(
                x_range,
                red_green + 0.4,
                color="#2ca25f",
                alpha=0.85,
                linewidth=1.6,
                label=label,
            )
            ax.plot(
                x_range, green_blue + 0.4, color="#2ca25f", alpha=0.85, linewidth=1.2
            )

        # Panel A: All galaxies
        contourf1 = create_density_contour_plot(
            df["logMsT"], df["g-i"], ax1, "All galaxies", "Panel A"
        )
        add_region_labels(ax1)
        add_separating_lines(ax1)

        # Panel B: Early-type galaxies
        early_type_df = df[df["morphology"] == "Early-type"]
        if len(early_type_df) > 50:
            create_density_contour_plot(
                early_type_df["logMsT"],
                early_type_df["g-i"],
                ax2,
                "Early-type galaxies",
                "Panel B",
            )
            add_separating_lines(ax2)
        else:
            ax2.text(
                0.5,
                0.5,
                "Insufficient data\nfor contours",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            ax2.set_title(
                "Panel B: Early-type galaxies",
                fontsize=TITLE_FONT_SIZE - 1,
                fontweight="bold",
            )

        # Panel C: Late-type galaxies
        late_type_df = df[df["morphology"] == "Late-type"]
        create_density_contour_plot(
            late_type_df["logMsT"],
            late_type_df["g-i"],
            ax3,
            "Late-type galaxies",
            "Panel C",
            add_legend_labels=True,
        )
        add_separating_lines(ax3, label="Green guides: empirical boundaries")

        # Set consistent axis properties
        mass_range = (8.5, 12.0)
        color_range = (0.0, 3.5)

        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(mass_range)
            ax.set_ylim(color_range)
            ax.set_xlabel(
                r"Stellar mass $\log_{10} M_{*}\,(M_\odot)$",
                fontsize=AXIS_LABEL_SIZE + 2,
            )
            ax.set_ylabel(
                r"$g-i$ colour (mag)",
                fontsize=AXIS_LABEL_SIZE + 2,
            )
            ax.grid(True, alpha=0.32)
            ax.set_facecolor("#fdfdfd")
            ax.tick_params(length=TICK_LENGTH, pad=TICK_PADDING)
            ax.margins(x=0.02, y=0.02)

        ax2.set_ylabel("")
        ax2.tick_params(labelleft=False)
        ax1.set_xlabel("")
        ax2.set_xlabel("")
        ax3.tick_params(axis="x", labelsize=TICK_LABEL_SIZE + 2)
        for axis in [ax1, ax2]:
            axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
        for axis in [ax1, ax2, ax3]:
            axis.tick_params(axis="y", labelsize=TICK_LABEL_SIZE + 2)

        # Add colorbar
        if contourf1:
            cbar = fig.colorbar(contourf1, cax=cbar_ax)
            cbar.set_label(
                DENSITY_BAR_LABEL,
                rotation=270,
                labelpad=16,
                fontsize=COLORBAR_LABEL_SIZE,
            )
            cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE, length=TICK_LENGTH)
        else:
            fig.delaxes(cbar_ax)

        legend_handles = [
            Line2D(
                [0],
                [0],
                color="#3f007d",
                linewidth=0.9,
                label="Density contours (outer envelope = low density)",
            ),
            Line2D(
                [0],
                [0],
                color="#2ca25f",
                linewidth=1.4,
                label="Green guides: red -> green -> blue transitions",
            ),
            Patch(
                facecolor=LOW_DENSITY_SHADE_COLOR,
                edgecolor="none",
                alpha=0.25,
                label="Shaded low-density region",
            ),
            Line2D(
                [0],
                [0],
                marker=ISOLATED_MARKER_STYLE["marker"],
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor=ISOLATED_MARKER_STYLE["edgecolors"],
                label="Isolated galaxies (<= {} per bin)".format(ISOLATION_THRESHOLD),
            ),
        ]
        _styled_legend(
            ax3,
            handles=legend_handles,
            loc="upper right",
            fontsize=LEGEND_FONT_SIZE + 2,
            handlelength=2.4,
            borderaxespad=1.0,
            bbox_to_anchor=(0.98, 0.98),
            frameon=True,
        )

        plot_path = os.path.join(plots_dir, "color_stellar_mass_diagram.png")
        plt.savefig(plot_path, format="png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Color-stellar mass diagram saved: {plot_path}")
        print(f"Total galaxies: {len(df)}")
        print(
            f"Early-type: {len(early_type_df)} ({len(early_type_df) / len(df) * 100:.1f}%)"
        )
        print(
            f"Late-type: {len(late_type_df)} ({len(late_type_df) / len(df) * 100:.1f}%)"
        )

    except Exception as e:
        print(f"Error generating color-stellar mass diagram: {e}")
