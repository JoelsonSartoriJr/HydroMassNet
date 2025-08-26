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
