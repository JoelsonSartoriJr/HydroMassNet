import logging
import os
import yaml

def setup_logging():
    """
    Configura o sistema de logging com base no arquivo config.yaml.
    Cria um logger raiz que escreve para um arquivo e para o console.
    """
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        log_config = config.get('logging', {})
        log_level = log_config.get('level', 'INFO').upper()
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = config.get('paths', {}).get('log_file', 'logs/pipeline.log')

        # Criar diretório de log se não existir
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Configurar o logger raiz
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler() # Envia logs para o console (stdout/stderr)
            ],
            force=True # Permite reconfigurar o logger
        )

        logging.info("="*50)
        logging.info("Sistema de Logging configurado com sucesso.")
        logging.info(f"Nível: {log_level}, Arquivo: {log_file}")
        logging.info("="*50)

    except Exception as e:
        # Fallback para configuração básica se houver erro
        logging.basicConfig(level=logging.INFO)
        logging.exception("Erro ao configurar o logging a partir do YAML. Usando configuração padrão.", exc_info=e)