import re

def parse_layers(layers_config_str: str, input_dim: int, output_dim: int = 1) -> list[int]:
    """
    Converte uma string como '128-64-32' em uma lista de dimensões
    [input_dim, 128, 64, 32, output_dim].
    """
    if not isinstance(layers_config_str, str) or not re.match(r'^(\d+)(-\d+)*$', layers_config_str):
        raise ValueError("O formato de 'layers' deve ser uma string com números separados por hífen, ex: '128-64'.")
    
    hidden_dims = [int(d) for d in layers_config_str.split('-')]
    return [input_dim] + hidden_dims + [output_dim]
