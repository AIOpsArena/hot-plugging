import yaml
import os
sys_config_file = 'config.yaml'

def get_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

sys_config = get_config(sys_config_file)

IGNORE_LIST = sys_config['ignore_list'].split()

def deal_config(config, user, base_path):
    new_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            new_config[k] = deal_config(v, user)
        else:
            if any(keyword in k for keyword in ['path', 'dir']):
                if k in IGNORE_LIST:
                    path = os.path.join(base_path, v)
                    if 'dir' in k:
                        if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)
                    new_config[k] = path
                elif v or v == '':
                    path = os.path.join(base_path, user, v)
                    if 'dir' in k:
                        if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)
                    new_config[k] = path
                else:
                    new_config[k] = v
            else:
                new_config[k] = v

    return new_config
