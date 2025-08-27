import tensorflow as tf

def setup_device():
    """Detecta e configura o dispositivo de hardware (GPU/CPU) para treinamento."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configura o crescimento de memória para evitar alocação total
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"--- Dispositivo: {len(gpus)} GPU(s) encontrada(s) e configurada(s). O treinamento usará a GPU. ---")
        except RuntimeError as e:
            # O crescimento de memória deve ser definido antes que as GPUs sejam inicializadas
            print(f"--- Erro ao configurar GPU: {e} ---")
            print("--- Dispositivo: Usando CPU. ---")
    else:
        print("--- Dispositivo: Nenhuma GPU encontrada. Usando CPU. ---")

def parse_layers(layers_str: str, input_dim: int, output_dim: int = 1):
    """
    Converte uma string como '256-128' para uma lista de dimensões de camada,
    incluindo as dimensões de entrada e saída.

    Parameters
    ----------
    layers_str : str
        String com as dimensões das camadas ocultas (ex: "256-128").
    input_dim : int
        Dimensão da camada de entrada.
    output_dim : int, optional
        Dimensão da camada de saída. O padrão é 1. Se None, não é adicionada.

    Returns
    -------
    list
        Lista com as dimensões de todas as camadas (ex: [10, 256, 128, 1]).
    """
    if not layers_str:
        hidden_units = []
    else:
        hidden_units = [int(u) for u in layers_str.split('-')]

    layer_dims = [input_dim] + hidden_units

    if output_dim is not None:
        layer_dims.append(output_dim)

    return layer_dims
