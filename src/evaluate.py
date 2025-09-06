# file: ./src/evaluate.py
import argparse
import yaml
import os
import json
import tensorflow as tf
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from src.data import DataHandler
from src.models.bayesian_dense_regression import BayesianDenseRegression
from src.models.bayesian_density_network import BayesianDensityNetwork
from src.models.vanilla_network import create_vanilla_model

def main():
    """
    Ponto de entrada para avaliar um modelo treinado.
    """
    parser = argparse.ArgumentParser(description="Script unificado para avaliação de modelos.")
    parser.add_argument('--model_type', type=str, required=True, choices=['bnn', 'dbnn', 'vanilla'])
    parser.add_argument('--model_config', type=str, required=True, help='String JSON com a configuração do modelo.')
    parser.add_argument('--weights_path', type=str, required=True, help='Caminho para os pesos do modelo treinado.')
    parser.add_argument('--output_dir', type=str, required=True, help='Diretório para salvar os artefatos de avaliação.')
    args = parser.parse_args()

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_cfg = json.loads(args.model_config)

    data_handler = DataHandler(config, feature_override=model_cfg.get('features'))
    _, _, _, _, x_test, y_test, features_out = data_handler.get_full_dataset_and_splits()
    input_shape = (len(features_out),)

    model = None
    if args.model_type == 'bnn':
        layers = [int(x) for x in model_cfg['layers'].split('-')]
        model = BayesianDenseRegression(layers, config)
    elif args.model_type == 'dbnn':
        core_layers = [int(x) for x in model_cfg['core_layers'].split('-')]
        head_layers = [int(x) for x in str(model_cfg['head_layers']).split('-')]
        model = BayesianDensityNetwork(core_layers, head_layers, config)
    elif args.model_type == 'vanilla':
        layers = [int(x) for x in model_cfg['layers'].split('-')]
        model = create_vanilla_model(layers, input_shape, model_cfg['dropout'])

    model(tf.constant(x_test[:1]))
    model.load_weights(args.weights_path)
    print(f"Pesos do modelo '{args.model_type}' carregados de {args.weights_path}")

    print("Avaliando o modelo no conjunto de teste...")
    predictions = model.predict(x_test, batch_size=model_cfg.get('batch_size', 128))

    if args.model_type == 'dbnn' and predictions.shape[1] > 1:
        predictions_for_metrics = predictions[:, 0]
    else:
        predictions_for_metrics = predictions

    mae = mean_absolute_error(y_test, predictions_for_metrics)
    r2 = r2_score(y_test, predictions_for_metrics)

    print("\n" + "="*50)
    print(f"Resultados da Avaliação para o Modelo: {args.model_type.upper()}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50 + "\n")

    base_name = f"champion_{args.model_type}"
    preds_df = pd.DataFrame({'true': y_test.values.flatten(), 'predicted': predictions_for_metrics.flatten()})
    preds_path = os.path.join(args.output_dir, f'{base_name}_predictions.csv')
    preds_df.to_csv(preds_path, index=False)
    print(f"Predições salvas em: {preds_path}")

    metrics_data = {'mae': mae, 'r2_score': r2}
    metrics_path = os.path.join(args.output_dir, f'{base_name}_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"Métricas de avaliação salvas em: {metrics_path}")

if __name__ == "__main__":
    main()
