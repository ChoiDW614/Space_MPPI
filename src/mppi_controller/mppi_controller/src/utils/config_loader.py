import os
import yaml

from ament_index_python.packages import get_package_share_directory


def load_config(package_name, config_name):
    config_path = os.path.join(get_package_share_directory(package_name), "configs", "canadarm", config_name)
    try:
        with open(config_path, "r") as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
            return params
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find config file '{config_name}' in package '{package_name}'"
            f"(expected at: {config_path})"
        ) from e
    return
