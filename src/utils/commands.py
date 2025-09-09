import subprocess
import logging

logger = logging.getLogger(__name__)

def run_command(command: list):
    """
    Executa um comando no shell, captura o output e verifica por erros.

    Args:
        command (list): Lista de strings representando o comando e seus argumentos.

    Raises:
        RuntimeError: Se o comando retornar um código de saída diferente de zero.

    Returns:
        str: A saída padrão (stdout) do comando executado.
    """
    command_str = ' '.join(command)
    logger.info(f"Executando comando: {command_str}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,  # Lança exceção automaticamente em caso de erro
            encoding='utf-8'
        )
        if result.stdout:
            logger.info(f"Comando bem-sucedido. STDOUT:\n{result.stdout.strip()}")
        return result.stdout

    except FileNotFoundError:
        logger.error(f"Erro: Comando '{command[0]}' não encontrado. Verifique se o programa está instalado e no PATH.")
        raise

    except subprocess.CalledProcessError as e:
        error_message = (
            f"Erro na execução do comando: {command_str}\n"
            f"Código de Retorno: {e.returncode}\n"
            f"STDOUT:\n{e.stdout.strip()}\n"
            f"STDERR:\n{e.stderr.strip()}"
        )
        logger.error(error_message)
        raise RuntimeError(error_message) from e