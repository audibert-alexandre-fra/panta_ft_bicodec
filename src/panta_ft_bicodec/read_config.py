import yaml


def read_config(filepath: str) -> dict:
    
    """
    Reads a YAML configuration file and returns its content as a dictionary.

    Args:
        filepath (str): Path to the YAML configuration file.

    Returns:
        dict: Content of the YAML configuration file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)  
    return config