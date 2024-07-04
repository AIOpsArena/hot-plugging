import os
import pathlib
import sys
root_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_path))
from algorithm_app.public_function import get_config, deal_config
import shutil

sys_config_file = 'config.yaml'
sys_config = get_config(sys_config_file)
base_dir = os.path.join(os.path.dirname(__file__), sys_config['experiment_dir'])

config_file = ''


class Algorithm:
    def __init__(self, user):
        config = get_config(config_file)
        config = deal_config(config, user, base_dir)
        self.config = config
        self.experiment_dir = os.path.join(base_dir, user)
        self.origin_metric_dir = os.path.join(base_dir, user, sys_config['origin_metric_dir'])
        self.origin_log_dir = os.path.join(base_dir, user, sys_config['origin_log_dir'])
        self.origin_trace_dir = os.path.join(base_dir, user, sys_config['origin_trace_dir'])
        self.origin_groundtruth_dir = os.path.join(base_dir, user, sys_config['origin_groundtruth_dir'])

    def run(self):
        try:
            pass
        except Exception as e:
            print(e)
        finally:
            self.clear()
    
    def clear(self):
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)
