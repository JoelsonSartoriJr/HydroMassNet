import os
import sys
import yaml
import json
import pandas as pd
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import logging
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logging_config import setup_logging
from src.utils.commands import run_command
setup_logging()
logger = logging.getLogger(__name__)

def run_hyperparameter_search():
    logger.info("Iniciando busca de hiperparâmetros.")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    search_config = config['hyperparameter_search']
    results_dir = config['paths']['search_results']
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Diretório de resultados: {results_dir}")

    full_report = []
    champion_configs = {}

    for model_type, params_grid in search_config.items():
        logger.info(f"\n{'='*80}\nIniciando busca para o modelo: {model_type.upper()}\n{'='*80}")
        
        grid = ParameterGrid(params_grid)
        best_mae = float('inf')
        best_params = None
        
        logger.info(f"Total de {len(grid)} combinações de hiperparâmetros para testar.")

        for i, params in enumerate(grid):
            logger.info(f"--- Testando {model_type.upper()} | Combinação {i+1}/{len(grid)}: {params} ---")
            
            unique_id = str(uuid.uuid4().hex)
            save_path = os.path.join(results_dir, f'temp_model_{model_type}_{unique_id}')

            try:
                process_output = run_command([
                    'poetry', 'run', 'python', '-m', 'src.train',
                    '--model_type', model_type,
                    '--model_config', json.dumps(params),
                    '--save_path', save_path
                ])
                
                performance_line = [line for line in process_output.strip().split('\n') if 'val_mae' in line][-1]
                performance = json.loads(performance_line)
                val_mae = performance['val_mae']
                
                logger.info(f"Resultado para {params}: val_mae = {val_mae:.4f}")
                
                params.update({'val_mae': val_mae, 'model_type': model_type})
                full_report.append(params)

                if val_mae < best_mae:
                    best_mae = val_mae
                    best_params = params
                    logger.info(f"*** Novo melhor resultado para {model_type.upper()}: {best_mae:.4f} ***")

            except (RuntimeError, IndexError, json.JSONDecodeError) as e:
                logger.error(f"Falha ao treinar com os parâmetros {params}. Erro: {e}", exc_info=False)
                params.update({'val_mae': float('inf'), 'model_type': model_type})
                full_report.append(params)
            finally:
                if os.path.exists(f"{save_path}.weights.h5"):
                    os.remove(f"{save_path}.weights.h5")

        if best_params:
            champion_key = f'champion_{model_type}'
            best_config = {k: v for k, v in best_params.items() if k not in ['val_mae', 'model_type']}
            champion_configs[champion_key] = best_config
            logger.info(f"Modelo campeão para {model_type.upper()}: MAE={best_mae:.4f}, Config={best_config}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f'full_search_report_{timestamp}.csv')
    pd.DataFrame(full_report).to_csv(report_path, index=False)
    logger.info(f"Relatório completo da busca salvo em: {report_path}")

    champion_path = os.path.join(results_dir, 'champion_config.yaml')
    with open(champion_path, 'w', encoding='utf-8') as f:
        yaml.dump(champion_configs, f, indent=2)
    logger.info(f"Configurações dos campeões salvas em: {champion_path}")

if __name__ == "__main__":
    run_hyperparameter_search()
