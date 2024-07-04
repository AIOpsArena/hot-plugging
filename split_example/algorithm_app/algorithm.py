# 此处import算法文件夹
import os
import pathlib
import sys
root_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_path))
from DiagFusion.DiagFusion_api import Diagfusion
from algorithm_app.public_function import get_config, deal_config
import shutil

sys_config_file = 'config.yaml'
sys_config = get_config(sys_config_file)
base_dir = os.path.join(os.path.dirname(__file__), sys_config['experiment_dir'])

# 需要用户填写
config_file = 'DiagFusion/config/platform_config.yaml'


class Algorithm:
    def __init__(self, user):
        config = get_config(config_file)
        config = deal_config(config, user, base_dir)
        self.config = config
        self.experiment_dir = os.path.join(base_dir, user)
        origin_metric_dir = os.path.join(base_dir, user, sys_config['origin_metric_dir'], 'container')
        origin_log_dir = os.path.join(base_dir, user, sys_config['origin_log_dir'])
        origin_trace_dir = os.path.join(base_dir, user, sys_config['origin_trace_dir'])
        origin_groundtruth_dir = os.path.join(base_dir, user, sys_config['origin_groundtruth_dir'])
        self.data_dir = {
            'origin_metric_dir': origin_metric_dir,
            'origin_log_dir': origin_log_dir,
            'origin_trace_dir': origin_trace_dir,
            'groundtruth_path': origin_groundtruth_dir
        }
        self.model = Diagfusion(config, self.data_dir)


    def train(self):
        try:
            self.model.preprocess_data()
            self.model.train_model()
        except Exception as e:
            print(e)
        finally:
            self.clear()

    def test(self):
        try:
            self.model.preprocess_data()
            result = self.model.test(test_type='anomaly')
            if result['status'] == 200:
                print(result['result'])
                return result['result']
        except Exception as e:
            print(e)
        finally:
            self.clear()
    
    def clear(self):
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)
