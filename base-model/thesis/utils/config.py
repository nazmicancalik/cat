import yaml
from easydict import EasyDict
from pathlib import Path
import os


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)

                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue

        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def get_config(cfg_path, logger=None):
    config = cfg_from_yaml_file(cfg_path)
    return config


def save_experiment_config(args, config, logger=None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))


def get_project_root() -> Path:
    return Path(__file__).parent.parent
