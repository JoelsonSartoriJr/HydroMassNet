import yaml
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from src.data import DataHandler

def main():
    """
    Treina e avalia um modelo de regressão linear como baseline.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    best_model_features = config['champion_models']['vanilla']['features']

    data_handler = DataHandler(config, feature_override=best_model_features)
    x_train, y_train, _, _, x_test, y_test, _ = data_handler.get_full_dataset_and_splits()

    print("--- Treinando e avaliando o modelo Baseline (Regressão Linear) ---")

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Baseline - R²: {r2:.4f}, RMSE: {rmse:.4f}")

    results_df = pd.DataFrame({
        'y_true': y_test.flatten(),
        'y_pred_mean': y_pred.flatten(),
        'y_pred_std': np.zeros_like(y_pred.flatten())
    })

    output_path = os.path.join(config['paths']['saved_models'], 'baseline_predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"Predições do Baseline salvas em: {output_path}")

if __name__ == "__main__":
    main()
