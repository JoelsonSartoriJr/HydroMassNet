import os
import sys
import yaml
import json
import logging
from datetime import datetime
from src.utils.commands import run_command
from src.utils.logging_config import setup_logging

# Configura o logging no início da execução
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """
    Ponto de entrada principal para orquestrar o pipeline completo.
    """
    logger.info(">>> INICIANDO O PIPELINE COMPLETO DE OTIMIZAÇÃO <<<")

    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Arquivo de configuração 'config.yaml' carregado.")

        # --- Passo 1: Seleção de Features ---
        logger.info("="*80)
        logger.info("PASSO 1: SELEÇÃO DOS MELHORES CONJUNTOS DE FEATURES")
        run_command(['poetry', 'run', 'python', '-m', 'src.feature_selection'])

        # --- Passo 2: Busca de Hiperparâmetros ---
        logger.info("="*80)
        logger.info("PASSO 2: BUSCA DE HIPERPARÂMETROS PARA CADA CONJUNTO DE FEATURES")
        run_command(['poetry', 'run', 'python', '-m', 'src.hyperparameter_search'])

        champion_config_path = os.path.join(config['paths']['search_results'], 'champion_config.yaml')
        if not os.path.exists(champion_config_path):
            logger.error("champion_config.yaml não foi encontrado. A busca de hiperparâmetros falhou.")
            raise FileNotFoundError("champion_config.yaml não foi encontrado.")

        with open(champion_config_path, 'r', encoding='utf-8') as f:
            champion_configs = yaml.safe_load(f)
        logger.info("Configurações dos modelos campeões carregadas.")

        # --- Passo 3: Treinamento e Avaliação dos Campeões ---
        logger.info("="*80)
        logger.info("PASSO 3: TREINAMENTO E AVALIAÇÃO DOS MODELOS CAMPEÕES")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        champions_dir = os.path.join(config['paths']['champion_results'], timestamp)
        os.makedirs(champions_dir, exist_ok=True)
        logger.info(f"Diretório de resultados dos campeões criado em: {champions_dir}")

        for name, model_cfg in champion_configs.items():
            model_type = name.replace('champion_', '')
            logger.info(f"Processando Campeão: {name.upper()}")
            save_path = os.path.join(champions_dir, f"champion_{model_type}")

            run_command([
                'poetry', 'run', 'python', '-m', 'src.train',
                '--model_type', model_type,
                '--model_config', json.dumps(model_cfg),
                '--save_path', save_path
            ])
            run_command([
                'poetry', 'run', 'python', '-m', 'src.evaluate',
                '--model_type', model_type,
                '--model_config', json.dumps(model_cfg),
                '--weights_path', f"{save_path}.weights.h5",
                '--output_dir', champions_dir
            ])

        # --- Passo 4: Geração de Gráficos ---
        logger.info("="*80)
        logger.info("PASSO 4: GERAÇÃO DOS GRÁFICOS FINAIS")
        run_command([
            'poetry', 'run', 'python', '-m', 'src.plotting',
            '--results_dir', champions_dir
        ])

        logger.info(">>> PIPELINE COMPLETO CONCLUÍDO COM SUCESSO! <<<")
        logger.info(f"Resultados finais disponíveis em: {champions_dir}")

    except FileNotFoundError as e:
        logger.critical(f"Erro de arquivo não encontrado: {e}", exc_info=True)
        sys.exit(1)
    except RuntimeError as e:
        logger.critical(f"Um passo do pipeline falhou: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Ocorreu um erro inesperado no pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()