import yaml


def get_config_data(path_to_config: str) -> dict:
    with open(path_to_config, 'r') as config_file:  # encoding = utf-8
        return yaml.load(config_file, Loader=yaml.FullLoader)
