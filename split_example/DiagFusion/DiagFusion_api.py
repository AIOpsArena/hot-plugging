import sys
import pytz
import pathlib
DF_path = pathlib.Path(__file__).parent
sys.path.append(str(DF_path))
root_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_path))
from process_data import process, process_trace, process_log, process_metric, create_run_table
import os
from transforms.events import fasttext_with_DA, sententce_embedding
from models import He_DGL
import pandas as pd
from datetime import *
class Diagfusion:
    def __init__(self, config, data_dir):
        self.config = config
        self.processed_data_config = config['processed_data']
        self.fasttext_config = config['fasttext']
        self.sentence_embedding_config = config['sentence_embedding']
        self.dgl_config = config['he_dgl']
        self.processed_data_config['origin_metric_dir'] = data_dir['origin_metric_dir']
        self.processed_data_config['origin_log_dir'] = data_dir['origin_log_dir']
        self.processed_data_config['origin_trace_dir'] = data_dir['origin_trace_dir']
        self.processed_data_config['groundtruth_path'] = data_dir['groundtruth_path']

    def preprocess_data(self):
        print('创建run_table')
        create_run_table(self.processed_data_config)
        print('run_table创建完成')

        print('指标预处理开始')
        process_metric(self.processed_data_config)
        print('指标预处理结束')

        print('日志预处理开始')
        process_log(self.processed_data_config)
        print('日志预处理结束')

        print('调用链预处理开始')
        process_trace(self.processed_data_config)
        print('调用链预处理结束')

        print('多模态数据整合开始')
        process(self.processed_data_config)
        print('多模态数据预处理结束')

    def train_model(self):
        action = 'train'
        print('[fasttext]')
        run_table = pd.read_csv(self.processed_data_config['run_table_path'])
        fasttext_with_DA.run_fasttext(self.fasttext_config, run_table, action)

        print('[sentence_embedding]')
        sententce_embedding.run_sentence_embedding(self.sentence_embedding_config)

        print('[dgl]')
        He_DGL.UnircaLab(self.dgl_config).do_lab(action)

    def test(self, test_type=None):
        action = 'test'
        print('[fasttext]')
        run_table = pd.read_csv(self.processed_data_config['run_table_path'])
        fasttext_with_DA.run_fasttext(self.fasttext_config, run_table, action)

        print('[sentence_embedding]')
        sententce_embedding.run_sentence_embedding(self.sentence_embedding_config)
        print('[dgl]')
        # 检查是否保存模型文件
        if not os.path.exists(self.dgl_config['service_model_path']) or not os.path.exists(
                self.dgl_config['failure_type_model_path']):
            return {'status': 500}
        result = He_DGL.UnircaLab(self.dgl_config).do_lab(action, test_type)
        return {'result': result, 'status': 200}