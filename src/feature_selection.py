import os
import sys
import yaml
import json
import pandas as pd
from itertools import combinations
from datetime import datetime
import logging

# Adiciona o diretório raiz ao path e configura logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logging_config import setup_logging
from src.utils.commands import run_command
setup_logging()
logger = logging.getLogger(__name__)

def run_feature_selection():
    """
    Executa a seleção de features treinando um modelo base com diferentes combinações.
    """
    logger.info("Iniciando processo de seleção de features.")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    fs_config = config['feature_selection']
    candidate_features = fs_config['candidate_features']
    min_features = fs_config['min_features']
    model_config = fs_config['evaluation_model_config']
    
    results_dir = config['paths']['feature_selection_results']
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Diretório de resultados: {results_dir}")

    all_combinations = []
    for r in range(min_features, len(candidate_features) + 1):
        all_combinations.extend(combinations(candidate_features, r))

    logger.info(f"Total de {len(all_combinations)} combinações de features a serem testadas.")
    
    results = []
    for i, combo in enumerate(all_combinations):
        features = list(combo)
        logger.info(f"--- Testando combinação {i+1}/{len(all_combinations)}: {features} ---")
        
        current_model_config = model_config.copy()
        current_model_config['features'] = features

        try:
            # O script `train.py` já está otimizado para GPU
            process_output = run_command([
                'poetry', 'run', 'python', '-m', 'src.train',
                '--model_type', 'vanilla',
                '--model_config', json.dumps(current_model_config),
                '--save_path', os.path.join(results_dir, 'temp_feature_model')
            ])
            
            # Extrai o JSON de performance da última linha do output
            performance_line = [line for line in process_output.strip().split('\n') if 'val_mae' in line][-1]
            performance = json.loads(performance_line)
            val_mae = performance['val_mae']
            
            logger.info(f"Resultado para {features}: val_mae = {val_mae:.4f}")
            results.append({
                'features': ', '.join(features),
                'n_features': len(features),
                'val_mae': val_mae
            })

        except (RuntimeError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Falha ao treinar com a combinação {features}. Erro: {e}", exc_info=True)
            results.append({
                'features': ', '.join(features),
                'n_features': len(features),
                'val_mae': float('inf') # Pior resultado possível para combinações que falharam
            })
        except Exception as e:
            logger.critical(f"Erro inesperado no processo de seleção de features: {e}", exc_info=True)
            # Decide se deve parar ou continuar
            continue 

    logger.info("Seleção de features concluída. Salvando relatório.")
    results_df = pd.DataFrame(results).sort_values(by='val_mae', ascending=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f'feature_selection_report_{timestamp}.csv')
    results_df.to_csv(report_path, index=False)
    
    logger.info(f"Relatório de seleção de features salvo em: {report_path}")
    logger.info(f"Melhor combinação encontrada:\n{results_df.iloc[0]}")

if __name__ == "__main__":
    run_feature_selection()