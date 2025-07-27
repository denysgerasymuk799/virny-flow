import yaml
import pathlib


def load_yaml_content():
    # Read and display the YAML config file
    yaml_path = pathlib.Path(__file__).parent.parent.parent.joinpath('scripts').joinpath('configs').joinpath('exp_config.yaml')
    try:
        with open(yaml_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: Could not find config file at {yaml_path}"
    except Exception as e:
        return f"Error reading config file: {str(e)}"


def load_yaml_config():
    """Load YAML configuration once during app startup"""
    try:
        yaml_path = pathlib.Path(__file__).parent.parent.parent.joinpath('scripts').joinpath('configs').joinpath('exp_config.yaml')
        with open(yaml_path, 'r') as file:
            return yaml.load(file, Loader=yaml.SafeLoader)
    except Exception as e:
        print(f"❌ Failed to load YAML config: {e}")
        return {}


def load_exp_config():
    """Load experimental configuration from YAML file"""
    config_path = pathlib.Path(__file__).parent.parent.parent.joinpath('scripts').joinpath('configs').joinpath('exp_config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
