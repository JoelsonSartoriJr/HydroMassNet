import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import glob
from sklearn.metrics import r2_score, mean_squared_error
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    "font.family": "serif", "text.usetex": False,
    "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "figure.dpi": 300,
    "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.6,
    "axes.formatter.use_mathtext": True
})

def plot_predictions_overview(predictions, config):
    """Generate individual plots of 'true vs predicted values' for each model."""
    sns.set_theme(style="whitegrid", rc=plt.rcParams)
    plots_dir = config['paths']['plots']

    # Get global min/max values for consistent scaling across all plots
    min_val, max_val = np.inf, -np.inf
    for _, df in predictions.items():
        min_val = min(min_val, df['y_true'].min(), df['y_pred_mean'].min())
        max_val = max(max_val, df['y_true'].max(), df['y_pred_mean'].max())

    padding = (max_val - min_val) * 0.05
    min_val -= padding
    max_val += padding

    for model_name, df in predictions.items():
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        y_true, y_pred = df['y_true'], df['y_pred_mean']
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Create hexbin plot
        hb = ax.hexbin(y_true, y_pred, gridsize=40, cmap='viridis', mincnt=1)
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax.set_title(f"{model_name.upper()}: $R^2={r2:.3f}$, RMSE={rmse:.3f}", fontsize=14)
        ax.set_xlabel(r'True Value: $\log_{10}(M_{\mathrm{HI}})$', fontsize=12)
        ax.set_ylabel(r'Predicted Value: $\log_{10}(M_{\mathrm{HI}})$', fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add colorbar
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Count', fontsize=12)

        plt.subplots_adjust(right=0.85)  # Make room for colorbar
        plot_path = os.path.join(plots_dir, f'{model_name}_predictions.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Prediction plot saved: {plot_path}")

def plot_confidence_intervals(predictions, config):
    """Plot individual predictions with confidence intervals for Bayesian models."""
    sns.set_theme(style="white", rc=plt.rcParams)
    bayesian_models = {k: v for k, v in predictions.items() if k in ['bnn', 'dbnn']}
    if not bayesian_models:
        print("No Bayesian models found for confidence interval plots.")
        return

    plots_dir = config['paths']['plots']

    for model_name, df in bayesian_models.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        df_sample = df.sample(n=min(len(df), 1000), random_state=config['seed'])
        df_sorted = df_sample.sort_values(by='y_true').reset_index()

        y_true, y_mean, y_std = df_sorted['y_true'], df_sorted['y_pred_mean'], df_sorted['y_pred_std']
        ci = 1.96 * y_std

        ax.fill_between(y_true, (y_mean - ci), (y_mean + ci), color='skyblue', alpha=0.5, label='95% Confidence Interval')
        ax.scatter(y_true, y_mean, color='navy', s=20, alpha=0.7, label='Mean Prediction')
        ax.plot(y_true, y_true, 'r--', linewidth=2, label='Identity Line')

        ax.set_title(f'{model_name.upper()}: Uncertainty Quantification', fontsize=14)
        ax.set_xlabel(r'True Value: $\log_{10}(M_{\mathrm{HI}})$', fontsize=12)
        ax.set_ylabel(r'Predicted Value: $\log_{10}(M_{\mathrm{HI}})$', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'{model_name}_confidence_intervals.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confidence interval plot saved: {plot_path}")

def plot_training_metrics(config):
    """Plot individual training metrics (loss and MAE) for each model."""
    results_dir = config['paths']['results']
    plots_dir = config['paths']['plots']

    # Find all history files
    history_files = glob.glob(os.path.join(results_dir, '*_history.csv'))

    if not history_files:
        print("No training history files found. Skipping training metrics plots.")
        return

    sns.set_theme(style="whitegrid", rc=plt.rcParams)

    for file_path in history_files:
        model_name = os.path.basename(file_path).replace('_history.csv', '').upper()
        try:
            df = pd.read_csv(file_path)
            epochs = range(1, len(df) + 1)

            # Create separate loss plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(epochs, df['loss'], label='Training Loss', linewidth=2, color='blue')
            if 'val_loss' in df.columns:
                ax.plot(epochs, df['val_loss'], label='Validation Loss', linewidth=2, linestyle='--', color='orange')

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{model_name}: Training Loss', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            plt.tight_layout()
            loss_plot_path = os.path.join(plots_dir, f'{model_name.lower()}_loss.pdf')
            plt.savefig(loss_plot_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Loss plot saved: {loss_plot_path}")

            # Create separate MAE plot
            if 'mae' in df.columns or 'val_mae' in df.columns:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                if 'mae' in df.columns:
                    ax.plot(epochs, df['mae'], label='Training MAE', linewidth=2, color='green')
                if 'val_mae' in df.columns:
                    ax.plot(epochs, df['val_mae'], label='Validation MAE', linewidth=2, linestyle='--', color='red')

                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
                ax.set_title(f'{model_name}: Training MAE', fontsize=14)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                mae_plot_path = os.path.join(plots_dir, f'{model_name.lower()}_mae.pdf')
                plt.savefig(mae_plot_path, format='pdf', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"MAE plot saved: {mae_plot_path}")

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

def plot_correlation_matrix(config):
    """Plot Pearson correlation matrix for the features used in models."""
    from src.hydromassnet.data import DataHandler

    plots_dir = config['paths']['plots']

    try:
        # Get all unique features from all models
        all_features = set()
        for model_name, model_config in config['models'].items():
            if 'features' in model_config:
                all_features.update(model_config['features'])

        if not all_features:
            print("No features found in model configurations.")
            return

        all_features = sorted(list(all_features))

        # Load data using the most comprehensive feature set
        data_handler = DataHandler(config, feature_override=all_features)
        x_train, y_train, x_val, y_val, x_test, y_test, features = data_handler.get_full_dataset_and_splits()

        # Combine all data for correlation analysis
        import pandas as pd
        all_data = pd.concat([
            pd.DataFrame(x_train, columns=features),
            pd.DataFrame(x_val, columns=features),
            pd.DataFrame(x_test, columns=features)
        ], ignore_index=True)

        # Calculate Pearson correlation matrix
        correlation_matrix = all_data.corr(method='pearson')

        # Create the plot
        sns.set_theme(style="white", rc=plt.rcParams)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Create a mask for the upper triangle (optional, for cleaner look)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        # Generate heatmap
        heatmap = sns.heatmap(correlation_matrix,
                             mask=mask,
                             annot=True,
                             cmap='RdBu_r',
                             center=0,
                             square=True,
                             fmt='.2f',
                             cbar_kws={'label': 'Pearson Correlation Coefficient ($r$)'},
                             annot_kws={'size': 8})

        ax.set_title('Pearson Correlation Coefficient ($r$)',
                    fontsize=12, pad=20)

        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'correlation_matrix.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Correlation matrix plot saved: {plot_path}")

    except Exception as e:
        print(f"Error generating correlation matrix: {e}")

def plot_correlation_matrix_complete(config):
    """Plot complete Pearson correlation matrix for the features used in models (no mask)."""
    from src.hydromassnet.data import DataHandler

    plots_dir = config['paths']['plots']

    try:
        # Get all unique features from all models
        all_features = set()
        for model_name, model_config in config['models'].items():
            if 'features' in model_config:
                all_features.update(model_config['features'])

        if not all_features:
            print("No features found in model configurations.")
            return

        all_features = sorted(list(all_features))

        # Load data using the most comprehensive feature set
        data_handler = DataHandler(config, feature_override=all_features)
        x_train, y_train, x_val, y_val, x_test, y_test, features = data_handler.get_full_dataset_and_splits()

        # Combine all data for correlation analysis
        import pandas as pd
        all_data = pd.concat([
            pd.DataFrame(x_train, columns=features),
            pd.DataFrame(x_val, columns=features),
            pd.DataFrame(x_test, columns=features)
        ], ignore_index=True)

        # Calculate Pearson correlation matrix
        correlation_matrix = all_data.corr(method='pearson')

        # Create the plot
        sns.set_theme(style="white", rc=plt.rcParams)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Generate complete heatmap (no mask)
        heatmap = sns.heatmap(correlation_matrix,
                             annot=True,
                             cmap='RdBu_r',
                             center=0,
                             square=True,
                             fmt='.2f',
                             cbar_kws={'label': 'Pearson Correlation Coefficient ($r$)'},
                             annot_kws={'size': 8},
                             linewidths=0.5,
                             linecolor='white')

        ax.set_title('Correlation Matrix (Pearson $r$) for Representative Predictors', fontsize=12, pad=20)

        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'correlation_matrix_complete.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Complete correlation matrix plot saved: {plot_path}")

    except Exception as e:
        print(f"Error generating complete correlation matrix: {e}")

def plot_feature_distributions(config):
    """Plot individual feature distributions to understand data characteristics."""
    from src.hydromassnet.data import DataHandler

    plots_dir = config['paths']['plots']

    try:
        # Get all unique features from all models
        all_features = set()
        for model_name, model_config in config['models'].items():
            if 'features' in model_config:
                all_features.update(model_config['features'])

        if not all_features:
            print("No features found in model configurations.")
            return

        all_features = sorted(list(all_features))

        # Load data
        data_handler = DataHandler(config, feature_override=all_features)
        x_train, y_train, x_val, y_val, x_test, y_test, features = data_handler.get_full_dataset_and_splits()

        # Combine all data
        import pandas as pd
        all_data = pd.concat([
            pd.DataFrame(x_train, columns=features),
            pd.DataFrame(x_val, columns=features),
            pd.DataFrame(x_test, columns=features)
        ], ignore_index=True)

        # Create individual distribution plots
        sns.set_theme(style="whitegrid", rc=plt.rcParams)

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
            ax.hist(all_data[feature], bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')

            # Add KDE curve
            from scipy import stats
            kde = stats.gaussian_kde(all_data[feature].dropna())
            x_range = np.linspace(all_data[feature].min(), all_data[feature].max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

            ax.set_xlabel(feature, fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'Distribution of {feature}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(features), rows * cols):
            row = i // cols
            col = i % cols
            if rows == 1:
                axes[col].set_visible(False)
            else:
                axes[row, col].set_visible(False)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, 'feature_distributions.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature distributions plot saved: {plot_path}")

    except Exception as e:
        print(f"Error generating feature distributions: {e}")

def plot_learning_curves(config):
    """Plot individual comprehensive learning curves for each model."""
    results_dir = config['paths']['results']
    plots_dir = config['paths']['plots']

    # Find all history files
    history_files = glob.glob(os.path.join(results_dir, '*_history.csv'))

    if not history_files:
        print("No training history files found. Skipping learning curves plots.")
        return

    sns.set_theme(style="whitegrid", rc=plt.rcParams)

    for file_path in history_files:
        model_name = os.path.basename(file_path).replace('_history.csv', '').upper()
        try:
            df = pd.read_csv(file_path)
            epochs = range(1, len(df) + 1)

            # Create combined learning curve with dual y-axis
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
            ax2 = ax1.twinx()

            # Plot Loss on left y-axis (log scale)
            line1 = ax1.plot(epochs, df['loss'], 'b-', label='Training Loss', linewidth=2)
            lines = line1
            if 'val_loss' in df.columns:
                line2 = ax1.plot(epochs, df['val_loss'], 'b--', label='Validation Loss', linewidth=2)
                lines.extend(line2)

            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss (log scale)', color='b', fontsize=12)
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)

            # Plot MAE on right y-axis
            if 'mae' in df.columns:
                line3 = ax2.plot(epochs, df['mae'], 'r-', label='Training MAE', linewidth=2)
                lines.extend(line3)
            if 'val_mae' in df.columns:
                line4 = ax2.plot(epochs, df['val_mae'], 'r--', label='Validation MAE', linewidth=2)
                lines.extend(line4)

            ax2.set_ylabel('Mean Absolute Error (MAE)', color='r', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='r')

            # Create combined legend
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper right', fontsize=11)

            ax1.set_title(f'{model_name} Model - Complete Learning Curve', fontsize=14)

            plt.tight_layout()
            learning_plot_path = os.path.join(plots_dir, f'{model_name.lower()}_learning_curve.pdf')
            plt.savefig(learning_plot_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Learning curve plot saved: {learning_plot_path}")

        except Exception as e:
            print(f"Error plotting learning curve for {file_path}: {e}")
            continue

def plot_all(predictions, config):
    """Main function to call all plotting routines."""
    plots_dir = config['paths']['plots']
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
    plots_dir = config['paths']['plots']

    try:
        # Load data
        df = pd.read_csv('data/hydromassnet_full_dataset_all_columns.csv')
        df = df.dropna(subset=['logMsT', 'g-i', 'b/a', 'logSFR22'])

        # Filter reasonable ranges
        df = df[(df['logMsT'] > 8.5) & (df['logMsT'] < 12.0)]
        df = df[(df['g-i'] > 0.0) & (df['g-i'] < 3.5)]

        # Create morphology classification
        ba_median = df['b/a'].median()
        sfr_median = df['logSFR22'].median()

        df['morphology'] = 'Late-type'
        early_type_mask = (df['b/a'] > ba_median) & (df['logSFR22'] < sfr_median)
        df.loc[early_type_mask, 'morphology'] = 'Early-type'

        # Create figure with 3 panels
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                             hspace=0.25, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])  # All galaxies
        ax2 = fig.add_subplot(gs[0, 1])  # Early-type
        ax3 = fig.add_subplot(gs[1, :])  # Late-type

        def create_density_contour_plot(x, y, ax, title, levels=10):
            """Create density contour plot for given data"""
            if len(x) < 10:
                ax.scatter(x, y, alpha=0.5, s=10)
                ax.set_title(title, fontweight='bold')
                return None

            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(x, y, bins=50)
            H = H.T
            X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
            H_smooth = gaussian_filter(H, sigma=1.0)

            # Plot filled contours and lines
            contourf = ax.contourf(X, Y, H_smooth, levels=levels, alpha=0.7,
                                  cmap='viridis_r', extend='max')
            ax.contour(X, Y, H_smooth, levels=levels, colors='brown',
                      alpha=0.6, linewidths=0.8)

            ax.set_title(title, fontweight='bold')
            return contourf

        def add_region_labels(ax):
            """Add red sequence, green valley, blue cloud labels"""
            ax.text(10.8, 2.7, 'red sequence', fontsize=12, color='red',
                    fontweight='bold', ha='center', style='italic')
            ax.text(10.2, 2.0, 'green valley', fontsize=12, color='green',
                    fontweight='bold', ha='center', style='italic')
            ax.text(9.8, 1.2, 'blue cloud', fontsize=12, color='blue',
                    fontweight='bold', ha='center', style='italic')

        def add_separating_lines(ax):
            """Add approximate separating lines between regions"""
            x_range = np.linspace(8.5, 12.0, 100)
            red_green = 0.8 + 0.15 * (x_range - 10.0)
            green_blue = 0.6 + 0.12 * (x_range - 10.0)
            ax.plot(x_range, red_green + 0.4, 'g-', alpha=0.8, linewidth=2)
            ax.plot(x_range, green_blue + 0.4, 'g-', alpha=0.8, linewidth=2)

        # Panel A: All galaxies
        contourf1 = create_density_contour_plot(df['logMsT'], df['g-i'], ax1,
                                               'All galaxies', levels=12)
        add_region_labels(ax1)
        add_separating_lines(ax1)

        # Panel B: Early-type galaxies
        early_type_df = df[df['morphology'] == 'Early-type']
        if len(early_type_df) > 50:
            create_density_contour_plot(early_type_df['logMsT'], early_type_df['g-i'],
                                       ax2, 'Early-type galaxies', levels=8)
            add_separating_lines(ax2)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor contours',
                    transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('Early-type galaxies', fontweight='bold')

        # Panel C: Late-type galaxies
        late_type_df = df[df['morphology'] == 'Late-type']
        create_density_contour_plot(late_type_df['logMsT'], late_type_df['g-i'],
                                   ax3, 'Late-type galaxies', levels=12)
        add_separating_lines(ax3)

        # Set consistent axis properties
        mass_range = (8.5, 12.0)
        color_range = (0.0, 3.5)

        axes = [ax1, ax2, ax3]
        for ax in axes:
            ax.set_xlim(mass_range)
            ax.set_ylim(color_range)
            ax.set_xlabel('Stellar Mass log M$_*$ (M$_☉$)')
            ax.set_ylabel('g - i Color (mag)')
            ax.grid(True, alpha=0.3)

        # Add colorbar
        if contourf1:
            cbar = fig.colorbar(contourf1, ax=axes, shrink=0.6, aspect=30)
            cbar.set_label('Galaxy Density', rotation=270, labelpad=20)

        plt.tight_layout()

        plot_path = os.path.join(plots_dir, 'color_stellar_mass_diagram.pdf')
        plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Color-stellar mass diagram saved: {plot_path}")
        print(f"Total galaxies: {len(df)}")
        print(f"Early-type: {len(early_type_df)} ({len(early_type_df)/len(df)*100:.1f}%)")
        print(f"Late-type: {len(late_type_df)} ({len(late_type_df)/len(df)*100:.1f}%)")

    except Exception as e:
        print(f"Error generating color-stellar mass diagram: {e}")
