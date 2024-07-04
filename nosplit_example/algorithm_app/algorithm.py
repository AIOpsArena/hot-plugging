# 此处import算法文件夹
import os
import pathlib
import sys
root_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_path))
from automap.automap import AutoMap
from algorithm_app.public_function import get_config, deal_config
import shutil

sys_config_file = 'config.yaml'
sys_config = get_config(sys_config_file)
base_dir = os.path.join(os.path.dirname(__file__), sys_config['experiment_dir'])

# 需要用户填写
config_file = 'automap/config.yaml'


class Algorithm:
    def __init__(self, user):
        config = get_config(config_file)
        config = deal_config(config, user, base_dir)
        self.config = config
        self.model = AutoMap(config)
        self.experiment_dir = os.path.join(base_dir, user)
        self.origin_metric_dir = os.path.join(base_dir, user, sys_config['origin_metric_dir'])
        self.origin_groundtruth_dir = os.path.join(base_dir, user, sys_config['origin_groundtruth_dir'])

    def train(self):
        pass

    def test(self):
        pass

    def run(self):
        try:
            print("In Algorithm.run")
            result = self.model.run(self.origin_metric_dir, self.origin_groundtruth_dir)
        except Exception as e:
            print(e)
        finally:
            self.clear()
            return result
    
    def clear(self):
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)
