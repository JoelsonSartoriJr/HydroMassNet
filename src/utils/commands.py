import subprocess

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
    print(f"--- Executando: {' '.join(command)} ---")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error_message = (
            f"### ERRO NA EXECUÇÃO ###\n"
            f"Comando: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        raise RuntimeError(error_message)
    print(result.stdout)
    return result.stdout
