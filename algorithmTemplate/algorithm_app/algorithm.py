# 此处import算法文件夹
import os
import pathlib
import sys
root_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_path))
from algorithm_app.public_function import get_config, deal_config
import shutil
"""
import your model here
"""
# import Model

sys_config_file = 'config.yaml'
sys_config = get_config(sys_config_file)
base_dir = os.path.join(os.path.dirname(__file__), sys_config['experiment_dir'])

"""
write the path of your config file
"""
config_file = 'algorithm/config.yaml'


class Algorithm:
    def __init__(self, user):
        config = get_config(config_file)
        config = deal_config(config, user, base_dir)

        """
        your need to initialize your model here
        """
        # self.model = Model(config)

        # this is the experiment dir
        self.experiment_dir = os.path.join(base_dir, user)

        # these are origin data path, use any kind of the data if you want
        self.origin_metric_dir = os.path.join(base_dir, user, sys_config['origin_metric_dir'])
        self.origin_log_dir = os.path.join(base_dir, user, sys_config['origin_log_dir'])
        self.origin_trace_dir = os.path.join(base_dir, user, sys_config['origin_trace_dir'])
        self.origin_groundtruth_dir = os.path.join(base_dir, user, sys_config['origin_groundtruth_dir'])

    def train(self):
        pass

    def test(self):
        pass

    def run(self):
        pass
    
    def clear(self):
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)


"""
format 1:
{
    'top@k': {
        0: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1},
        1: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1},
        2: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1}
    },
    'epoch': 3
}

format 2:
{
    'accuracy@k': {
        0: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1},
        1: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1},
        2: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1}
    },
    'epoch': 3
}

format 3:
{
    'prf': {
        0: {'precision': 0.1, 'recall': 0.1, 'f1-score': 0.1},
        1: {'precision': 0.1, 'recall': 0.1, 'f1-score': 0.1},
        2: {'precision': 0.1, 'recall': 0.1, 'f1-score': 0.1}
    },
    'epoch': 3
}
"""