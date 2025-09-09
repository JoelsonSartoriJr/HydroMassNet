# file: ./src/feature_selection.py
"""
Executes a feature selection process by training models with different feature combinations in parallel.
"""
import os
import sys
import yaml
import json
import pandas as pd
from itertools import combinations
from datetime import datetime
import logging
import uuid
import multiprocessing
from functools import partial

# Adiciona o diretório raiz ao path para importação de módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logging_config import setup_logging
from src.utils.commands import run_command

# Configuração inicial do logging
setup_logging()
logger = logging.getLogger(__name__)

def train_feature_combination(features, model_config, results_dir, total_combinations, i):
    """
    Worker function to train and evaluate a single combination of features.
    This function is designed to be executed in a separate process.
    """
    log_prefix = f"[Combinação {i+1}/{total_combinations}]"
    logger.info(f"{log_prefix} Testando features: {features}")

    current_model_config = model_config.copy()
    current_model_config['features'] = features
    
    # Garante que cada processo tenha um nome de ficheiro único para evitar conflitos
    unique_id = str(uuid.uuid4().hex)
    save_path = os.path.join(results_dir, f'temp_model_{unique_id}')
    
    result = {
        'features': ', '.join(features), 
        'n_features': len(features), 
        'val_mae': float('inf')
    }

    try:
        command = [
            'poetry', 'run', 'python', '-m', 'src.train',
            '--model_type', 'vanilla',
            '--model_config', json.dumps(current_model_config),
            '--save_path', save_path
        ]
        
        # A função run_command executa o processo de treino
        process_output = run_command(command)
        
        # Extrai a última linha JSON com as métricas de validação
        performance_line = [line for line in process_output.strip().split('\n') if 'val_mae' in line][-1]
        performance = json.loads(performance_line)
        val_mae = performance['val_mae']
        
        logger.info(f"{log_prefix} Resultado: val_mae = {val_mae:.4f}")
        result['val_mae'] = val_mae

    except (RuntimeError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"{log_prefix} Falha ao treinar com a combinação {features}. Erro: {e}", exc_info=False)
    
    finally:
        # Garante a limpeza do ficheiro de pesos temporário
        weights_file = f"{save_path}.weights.h5"
        if os.path.exists(weights_file):
            try:
                os.remove(weights_file)
            except OSError as e:
                logger.error(f"Erro ao remover o ficheiro temporário {weights_file}: {e}")
                
    return result

def run_feature_selection():
    """
    Main function to orchestrate the feature selection process using a pool of workers.
    """
    logger.info("Iniciando processo de seleção de features.")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    fs_config = config['feature_selection']
    candidate_features = fs_config['candidate_features']
    min_features = fs_config['min_features']
    model_config = fs_config['evaluation_model_config']
    
    # Lê o número de jobs paralelos do config.yaml, com um valor padrão seguro
    num_parallel_jobs = fs_config.get('num_parallel_jobs', max(1, multiprocessing.cpu_count() // 2))
    logger.info(f"Executando com {num_parallel_jobs} processos paralelos.")
    
    results_dir = config['paths']['feature_selection_results']
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Diretório de resultados: {results_dir}")

    # Gera todas as combinações de features
    all_combinations = [
        list(c) for r in range(min_features, len(candidate_features) + 1) 
        for c in combinations(candidate_features, r)
    ]
    total_combinations = len(all_combinations)
    logger.info(f"Total de {total_combinations} combinações de features a serem testadas.")
    
    # Prepara os argumentos para cada tarefa
    tasks = [
        (features, i) for i, features in enumerate(all_combinations)
    ]

    # Cria uma função parcial com os argumentos que são constantes para todas as tarefas
    worker_func = partial(train_feature_combination, 
                          model_config=model_config, 
                          results_dir=results_dir, 
                          total_combinations=total_combinations)

    # Usa um Pool de processos para executar os treinamentos em paralelo
    with multiprocessing.Pool(processes=num_parallel_jobs) as pool:
        # imap_unordered processa as tarefas à medida que são submetidas e retorna os resultados
        # à medida que ficam prontos, o que é eficiente para logging e progresso.
        results = list(pool.starmap(worker_func, tasks))

    logger.info("Seleção de features concluída. Salvando relatório.")
    results_df = pd.DataFrame(results).sort_values(by='val_mae', ascending=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(results_dir, f'feature_selection_report_{timestamp}.csv')
    results_df.to_csv(report_path, index=False)
    
    logger.info(f"Relatório de seleção de features salvo em: {report_path}")
    logger.info(f"Melhor combinação encontrada:\n{results_df.iloc[0]}")

if __name__ == "__main__":
    # Garante que o pool de processos funcione corretamente em diferentes SOs
    multiprocessing.set_start_method('spawn', force=True)
    run_feature_selection()
