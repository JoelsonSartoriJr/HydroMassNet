import os
import subprocess
import yaml

def main():
    """
    Lê a configuração do modelo campeão e inicia o treinamento final.
    """
    print("#"*80)
    print("### INICIANDO TREINAMENTO DO MODELO CAMPEÃO ###")
    print("#"*80)

    # Carrega a configuração principal para obter os caminhos
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    champion_config_path = os.path.join(config['paths']['search_results'], 'champion_config.yaml')

    if not os.path.exists(champion_config_path):
        print(f"ERRO: Arquivo de configuração do campeão não encontrado em '{champion_config_path}'")
        print("Execute o pipeline de otimização completa primeiro.")
        return

    with open(champion_config_path, 'r') as f:
        champion_model_name = yaml.safe_load(f)['champion_model']['model']

    print(f"Modelo campeão identificado: {champion_model_name.upper()}")

    # Constrói e executa o comando para treinar o modelo final
    command = [
        'python', '-m', 'src.train_final',
        '--config_path', champion_config_path
    ]

    print(f"Executando comando: {' '.join(command)}")

    # O PWD (diretório de trabalho atual) já é a raiz do projeto
    subprocess.run(command, check=True)

    print("\n" + "#"*80)
    print("### TREINAMENTO DO MODELO CAMPEÃO CONCLUÍDO! ###")
    print("O próximo passo é avaliar o modelo no conjunto de teste.")
    print("#"*80)


if __name__ == "__main__":
    main()
