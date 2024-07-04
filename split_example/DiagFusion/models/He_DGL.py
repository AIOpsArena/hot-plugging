import os
import random
from tqdm import tqdm
from torch import tensor
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import dgl
print(dgl.__version__, dgl.__path__)
import dgl.data.utils as U
import time
import pickle
from .layers import *
from sklearn.metrics import precision_score, f1_score, recall_score
import warnings
import sys
import pathlib
DF_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(DF_path))

warnings.filterwarnings('ignore')


class UnircaDataset:
    """
    参数
    ----------
    dataset_path: str
        数据存放位置。
        举例: 'train_Xs.pkl' （67 * 14 * 40）（图数 * 节点数 * 节点向量维数）
    labels_path: str
        标签存放位置。
        举例: 'train_ys_anomaly_type.pkl' （67）
    topology: str
        图的拓扑结构存放位置
        举例：'topology.pkl'
    aug: boolean (default: False)
        需要数据增强，该值设置为True
    aug_size: int (default: 0)
        数据增强时，每个label对应的样本数
    shuffle: boolean (default: False)
        load()完成以后，若shuffle为True，则打乱self.graphs 和 self.labels （同步）
    """

    def __init__(self, dataset_path, labels_path, topology, aug=False, aug_size=0, shuffle=False):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.topology = topology
        self.aug = aug
        self.aug_size = aug_size
        self.graphs = []
        self.labels = []
        self.load()
        if shuffle:
            self.shuffle()

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)

    def load(self):
        """ __init__()  中使用，作用是装载 self.graphs 和 self.labels，若aug为True，则进行数据增强操作。
        """
        Xs = tensor(U.load_info(self.dataset_path))
        ys = tensor(U.load_info(self.labels_path))
        topology = U.load_info(self.topology)
        assert Xs.shape[0] == ys.shape[0]
        if self.aug:
            Xs, ys = self.aug_data(Xs, ys)

        for X in Xs:
            g = dgl.graph(topology)  # 同质图
            # 若有0入度节点，给这些节点加自环
            in_degrees = g.in_degrees()
            zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
            for node in zero_indegree_nodes:
                g.add_edges(node, node)

            g.ndata['attr'] = X
            self.graphs.append(g)
        self.labels = ys

    def shuffle(self):
        graphs_labels = [(g, l) for g, l in zip(self.graphs, self.labels)]
        random.shuffle(graphs_labels)
        self.graphs = [i[0] for i in graphs_labels]
        self.labels = [i[1] for i in graphs_labels]

    def aug_data(self, Xs, ys):
        """ load() 中使用，作用是数据增强
        参数
        ----------
        Xs: tensor
            多个图对应的特征向量矩阵。
            举例：67个图对应的Xs规模为 67 * 14 * 40 （67个图，每个图14个节点）
        ys: tensor
            每个图对应的label，要求是从0开始的整数。
            举例：如果一共有10个label，那么ys中元素值为 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        self.aug_size: int
            数据增强时，每个label对应的样本数

        返回值
        ----------
        aug_Xs: tensor
            数据增强的结果
        aug_ys: tensor
            数据增强的结果
        """
        aug_Xs = []
        aug_ys = []
        num_label = len(set([y.item() for y in ys]))
        grouped_Xs = [[] for i in range(num_label)]
        for X, y in zip(Xs, ys):
            grouped_Xs[y.item()].append(X)
        for group_idx in range(len(grouped_Xs)):
            cur_Xs = grouped_Xs[group_idx]
            n = len(cur_Xs)
            m = Xs.shape[1]
            while len(cur_Xs) < self.aug_size:
                select = np.random.choice(n, m)
                aug_X = torch.zeros_like(Xs[0])
                for i, j in zip(select, range(m)):
                    aug_X[j] = cur_Xs[i][j].detach().clone()
                cur_Xs.append(aug_X)
            for X in cur_Xs:
                aug_Xs.append(X)
                aug_ys.append(group_idx)
        aug_Xs = torch.stack(aug_Xs, 0)
        aug_ys = tensor(aug_ys)
        return aug_Xs, aug_ys


class RawDataProcess:
    """用来处理原始数据的类
    参数
    ----------
    config: dict
        配置参数
        Xs: 多个图的特征向量矩阵
        data_dir: 数据和结果存放路径
        dataset: 数据集名称 可选['21aiops', 'gaia']
    """

    def __init__(self, config):
        self.config = config

    def process(self):
        """ 用来获取并保存中间数据
        输入：
            sentence_embedding.pkl
            demo.csv
        输出：
            训练集：
                train_Xs.pkl
                train_ys_anomaly_type.pkl
                train_ys_service.pkl
            拓扑：
                topology.pkl
        """
        run_table = pd.read_csv(self.config['run_table_path'])
        Xs = U.load_info(os.path.join(self.config['data_dir'], self.config['Xs']))
        Xs = np.array(Xs)
        label_types = ['failure_type', 'service']
        label_dict = {label_type: None for label_type in label_types}
        for label_type in label_types:
            label_dict[label_type] = self.get_label(label_type, self.config[label_type].split(), run_table)
        save_dir = self.config['save_dir']
        #         train_size = self.config['train_size']
        train_index = run_table.index
        # 保存特征向量，特征向量是先训练集后测试集
        U.save_info(os.path.join(save_dir, 'train_Xs.pkl'), Xs[train_index])
        # 保存标签
        for label_type, labels in label_dict.items():
            U.save_info(os.path.join(save_dir, f'train_ys_{label_type}.pkl'), labels[train_index])
        # 保存拓扑
        topology = self.get_topology()
        U.save_info(os.path.join(save_dir, 'topology.pkl'), topology)
        # 保存边的类型(异质图)
        if self.config['heterogeneous']:
            edge_types = self.get_edge_types()
            U.save_info(os.path.join(save_dir, 'edge_types.pkl'), edge_types)

    def get_label(self, label_type, type_list, run_table):
        """ process() 中调用，用来获取label
        参数
        ----------
        label_type: str
            label的类型，可选：['service', 'anomaly_type']
        run_table: pd.DataFrame

        返回值
        ----------
        labels: torch.tensor()
            label列表
        """
        meta_labels = sorted(list(set(list(type_list))))
        labels_idx = {label: idx for label, idx in zip(meta_labels, range(len(meta_labels)))}
        print('label_type', label_type, 'label_idx', labels_idx)
        with open(os.path.join(self.config['model_dir'], f'{label_type}_idx.pkl'), 'wb') as file:
            pickle.dump(labels_idx, file)
        labels = np.array(run_table[label_type].apply(lambda label_str: labels_idx[label_str]))
        return labels

    def get_topology(self):
        """ process() 中调用，用来获取topology
        """
        dataset = self.config['dataset']
        if self.config['heterogeneous']:
            # 异质图
            if dataset == '22aiops':
                topology = (
                    [1, 1, 1, 1, 7, 1, 1, 4, 4, 4, 4, 4, 0, 3, 6, 9, 6, 5, 2, 1, 2, 0, 6, 7, 0, 1, 0, 2, 0, 3, 0, 4, 0,
                     5, 0, 6, 0, 7, 0, 8, 0, 9, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 2, 3, 2, 4, 2, 5, 2, 6,
                     2, 7, 2, 8, 2, 9, 3, 4, 3, 5, 3, 6, 3, 7, 3, 8, 3, 9, 4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 5, 6, 5, 7, 5,
                     8, 5, 9, 6, 7, 6, 8, 6, 9, 7, 8, 7, 9, 8, 9],
                    [0, 3, 6, 9, 6, 5, 2, 1, 2, 0, 6, 7, 1, 1, 1, 1, 7, 1, 1, 4, 4, 4, 4, 4, 1, 0, 2, 0, 3, 0, 4, 0, 5,
                     0, 6, 0, 7, 0, 8, 0, 9, 0, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 3, 2, 4, 2, 5, 2, 6, 2,
                     7, 2, 8, 2, 9, 2, 4, 3, 5, 3, 6, 3, 7, 3, 8, 3, 9, 3, 5, 4, 6, 4, 7, 4, 8, 4, 9, 4, 6, 5, 7, 5, 8,
                     5, 9, 5, 7, 6, 8, 6, 9, 6, 8, 7, 9, 7, 9, 8])
            else:
                raise Exception()
        else:
            # 同质图
            if dataset == '22aiops':
                topology = (
                    [1, 1, 1, 1, 7, 1, 1, 4, 4, 4, 4, 4, 1, 0, 2, 0, 3, 0, 4, 0,
                     5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 2, 3, 2, 4, 2, 5, 2, 6,
                     2, 7, 2, 8, 2, 9, 3, 4, 3, 5, 3, 6, 3, 7, 3, 8, 3, 9, 4, 5, 4, 6, 4, 7, 4, 8, 4, 9, 5, 6, 5, 7, 5, 8, 
                     5, 9, 6, 7, 6, 8, 6, 9, 7, 8, 7, 9, 8, 9],
                    [0, 3, 6, 9, 6, 5, 2, 1, 2, 0, 6, 7, 0, 1, 0, 2, 0, 3, 0, 4,
                     0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 3, 2, 4, 2, 5, 2, 6, 2,
                     7, 2, 8, 2, 9, 2, 4, 3, 5, 3, 6, 3, 7, 3, 8, 3, 9, 3, 5, 4, 6, 4, 7, 4, 8, 4, 9, 4, 6, 5, 7, 5, 8, 5,
                     9, 5, 7, 6, 8, 6, 9, 6, 8, 7, 9, 7, 9, 8])  # 正向
            else:
                raise Exception()
        return topology

    def get_edge_types(self):
        dataset = self.config['dataset']
        if not self.config['heterogeneous']:
            raise Exception()
        if dataset == '22aiops':
            etype = tensor(np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).astype(np.int64))
        else:
            raise Exception()
        return etype


class UnircaLab:
    def __init__(self, config):
        self.config = config
        instances = config['nodes'].split()
        self.ins_dict = dict(zip(instances, range(len(instances))))
        with open(os.path.join(self.config['model_dir'], 'instance_idx.pkl'), 'wb') as file:
            pickle.dump(self.ins_dict, file)
        self.demos = pd.read_csv(self.config['run_table_path'])
        if config['dataset'] == '22aiops':
            # self.topoinfo = {0: [8], 1: [0, 2, 3, 5, 6, 9], 2: [], 3: [], 4: [0, 1, 2, 6, 7], 5: [], 6: [], 7: [6],
            #                  8: [], 9: []}
            self.topoinfo = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6], 7: [7],
                             8: [8], 9: [9]}
        else:
            raise Exception('Unknow dataset')
        self.idx2instance = dict(zip(range(len(instances)), instances))
        services = self.config["service"].split()
        services.sort()
        self.services = services
        anomalys = [a.lstrip(' ') for a in self.config["failure_type"].split() if a != '' and a != ' ']
        anomalys.sort()
        self.anomalys = anomalys
        self.service_dict = dict(zip(services, range(len(services))))
        self.idx2service = dict(zip(range(len(services)), services))
        self.idx2anomaly = dict(zip(range(len(anomalys)), anomalys))
        print('idx2service', self.idx2service)
        print('idx2instance', self.idx2instance)
        print('idx2anomaly', self.idx2anomaly)

    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

    def save_result(self, save_path, data):
        # df = pd.DataFrame(data, columns=['top_k', 'accuracy'])
        df = data.melt(var_name='top_k', value_name='accuracy')
        print(df)
        df.to_csv(save_path, index=False)

    def multi_trainv0(self, dataset_ts, dataset_ta):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        weight = 0.5
        device = 'cpu'
        dataloader_ts = DataLoader(dataset_ts, batch_size=self.config['batch_size'], collate_fn=self.collate)
        dataloader_ta = DataLoader(dataset_ta, batch_size=self.config['batch_size'], collate_fn=self.collate)
        in_dim_ts = dataset_ts.graphs[0].ndata['attr'].shape[1]
        out_dim_ts = len(self.services)
        hid_dim_ts = (in_dim_ts + out_dim_ts) * 2 // 3
        in_dim_ta = dataset_ta.graphs[0].ndata['attr'].shape[1]
        out_dim_ta = len(self.anomalys)
        hid_dim_ta = (in_dim_ta + out_dim_ta) * 2 // 3
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model_ts = RGCNClassifier(in_dim_ts, hid_dim_ts, out_dim_ts, etype).to(device)
            model_ta = RGCNClassifier(in_dim_ta, hid_dim_ta, out_dim_ta, etype).to(device)
        else:
            model_ts = TAGClassifier(in_dim_ts, hid_dim_ts, out_dim_ts).to(device)
            model_ta = TAGClassifier(in_dim_ta, hid_dim_ta, out_dim_ta).to(device)

        print(model_ts)
        print(model_ta)

        opt_ts = torch.optim.Adam(model_ts.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        opt_ta = torch.optim.Adam(model_ta.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model_ts.train()
        model_ta.train()

        ts_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ts]
        ta_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ta]
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for i in range(len(ts_samples)):
                # service
                ts_bg = ts_samples[i][0].to(device)
                ts_labels = ts_samples[i][1].to(device)
                ts_feats = ts_bg.ndata['attr'].float()
                ts_logits = model_ts(ts_bg, ts_feats)
                ts_loss = F.cross_entropy(ts_logits, ts_labels)
                # anomaly_type
                ta_bg = ta_samples[i][0].to(device)
                ta_labels = ta_samples[i][1].to(device)
                ta_feats = ta_bg.ndata['attr'].float()
                ta_logits = model_ta(ta_bg, ta_feats)
                ta_loss = F.cross_entropy(ta_logits, ta_labels)

                opt_ts.zero_grad()
                opt_ta.zero_grad()

                total_loss = weight * ts_loss + (1 - weight) * ta_loss
                total_loss.backward()
                opt_ts.step()
                opt_ta.step()
                epoch_loss += total_loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                break
        return model_ts, model_ta

    def testv2(self, model, dataset, task, out_file, save_file=None):
        model.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset) + 10, collate_fn=self.collate)
        device = 'cpu'
        accuracy = []
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            output = model(batched_graph, batched_graph.ndata['attr'].float())
            k = 5 if output.shape[-1] >= 5 else output.shape[-1]
            if task == 'instance':
                _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)
                out_dir = os.path.join(self.config['save_dir'], 'preds')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                y_pred = indices.detach().numpy()
                y_true = labels.detach().numpy().reshape(-1, 1)
                ser_res = pd.DataFrame(np.append(y_pred, y_true, axis=1),
                                       columns=np.append([f'Top{i}' for i in range(1, len(y_pred[0]) + 1)],
                                                         'GroundTruth'))

                # 定位到实例级别
                ins_res, ins_failure_res = self.test_instance_local(ser_res, max_num=5)
                ins_res.to_csv(f'{out_dir}/{out_file}')
            elif task == 'failure_type':
                _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)
                out_dir = os.path.join(self.config['save_dir'], 'preds')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                y_pred = indices.detach().numpy()
                y_true = labels.detach().numpy().reshape(-1, 1)
                
                test_cases = self.demos
                failure_res = pd.DataFrame(np.append(y_pred, y_true, axis=1),
                                           columns=np.append([f'Top{i}' for i in range(1, len(y_pred[0]) + 1)],
                                                             'GroundTruth'),
                                           index=test_cases.index)
                failure_res.to_csv(f'{out_dir}/{out_file}')
            else:
                raise Exception('Unknow task')
        return output, labels

    def test_instance_local(self, s_preds, max_num=2):
        """
        根据微服务的预测结果预测微服务的根因实例
        """
        with open(self.config['text_path'], 'rb') as f:
            info = pickle.load(f)
        ktype = type(list(info.keys())[0])
        test_cases = self.demos
        ins_preds = []
        ins_failures = []
        i = 0
        for index, row in test_cases.iterrows():
            index = ktype(index)
            num_dict = {}
            for pair in info[index]:
                num_dict[self.ins_dict[pair[0]]] = len(info[index][pair].split())
            s_pred = s_preds.loc[i]
            ins_pred = []
            ins_failure = []
            for col in list(s_preds.columns)[: -1]:
                temp = sorted([(ins_id, num_dict[ins_id]) for ins_id in self.topoinfo[s_pred[col]]],
                              key=lambda x: x[-1], reverse=True)
                ins_pred.extend([item[0] for item in temp[: max_num]])
                # ins_failure.extend([item[1] for item in temp[: max_num]])
                ins_failure.extend([s_pred[col] for item in temp[: max_num]])
            ins_preds.append(ins_pred[: 5])
            ins_failures.append(ins_failure[: 5])
            i += 1
        # ins_preds 里没有5个的话会报错
        for i in range(len(ins_preds)):
            if len(ins_preds[i]) < 5:
                for j in range(5 - len(ins_preds[i])):
                    ins_preds[i].append(-1)
                    ins_failures[i].append(-1)

        y_true = np.array([self.ins_dict[ins] for ins in test_cases['instance'].values]).reshape(-1, 1)

        ins_res = pd.DataFrame(np.append(
            ins_preds, y_true, axis=1), columns=[
            'Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'GroundTruth'], index=test_cases.index)
        ins_failure_res = pd.DataFrame(ins_failures, columns=[
            'Top1', 'Top2', 'Top3', 'Top4', 'Top5'], index=test_cases.index)
        return ins_res, ins_failure_res

    def cross_evaluate(self, s_output, s_labels, a_output, a_labels, save_file=None):
        N_S = len(self.services)
        N_A = len(self.anomalys)
        TOPK_SA = self.config['TOPK_SA']
        # softmax取正（使用笛卡尔积比大小）
        s_values = nn.Softmax(dim=1)(s_output)
        a_values = nn.Softmax(dim=1)(a_output)
        # 获得 K_ * K_的笛卡尔积
        product = []
        for k in range(len(s_values)):
            service_val = s_values[k]
            anomaly_val = a_values[k]
            m = torch.zeros(N_S * N_A).reshape(N_S, N_A)
            for i in range(N_S):
                for j in range(N_A):
                    m[i][j] = service_val[i] * anomaly_val[j]
            product.append(m)
        # 获得每个笛卡尔积矩阵的topk及坐标
        sa_topks = []
        for idx in range(len(product)):
            m = product[idx]
            topk = []
            last_max_val = 1
            for k in range(TOPK_SA):
                cur_max_val = tensor(0)
                x = 0
                y = 0
                for i in range(N_S):
                    for j in range(N_A):
                        if m[i][j] > cur_max_val and m[i][j] < last_max_val:
                            cur_max_val = m[i][j]
                            x = i
                            y = j
                topk.append(((x, y), cur_max_val.item()))
                last_max_val = cur_max_val
            sa_topks.append(topk)

        # 使用笛卡尔积计算分数得到service + anomaly_type 的topk结果
        accuracy = []
        for k in range(1, TOPK_SA + 1):
            num = 0
            for i in range(len(s_labels)):
                label = (s_labels[i].item(), a_labels[i].item())
                predicts = sa_topks[i][:k]
                for predict in predicts:
                    if predict[0] == label:
                        num += 1
                        break
            print(f'top{k} acc: ', num / len(s_labels))
            accuracy.append([k, num / len(s_labels)])
        if save_file:
            seed = self.config['seed']
            save_dir = os.path.join(self.config['save_dir'], 'evaluations', 'service_anomaly')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_result(f'{save_dir}/seed{seed}_{save_file}', accuracy)

    def calc_instance_anomaly_tuple(self, s_output, s_labels, a_output, a_labels):
        N_S = len(self.services)
        N_A = len(self.anomalys)
        print('N_S, N_A', N_S, N_A)
        TOPK_SA = self.config['TOPK_SA']
        with open(self.config['text_path'], 'rb') as f:
            info = pickle.load(f)
        test_cases = self.demos
        ins_event_num_list = []
        i = 0
        for index, row in test_cases.iterrows():
            num_dict = {}
            for pair in info[index]:
                num_dict[self.ins_dict[pair[0]]] = len(info[index][pair].split())
            ins_event_num_list.append((index, num_dict))
        # softmax取正（使用笛卡尔积比大小）
        s_values = nn.Softmax(dim=1)(s_output)
        a_values = nn.Softmax(dim=1)(a_output)
        print('s_values', s_values)
        print('a_values', a_values)
        # 获得 K_ * K_的笛卡尔积
        product = []
        for k in range(len(s_values)):
            service_val = s_values[k]
            anomaly_val = a_values[k]
            m = torch.zeros(N_S * N_A).reshape(N_S, N_A)
            print('m', m)
            for i in range(N_S):
                for j in range(N_A):
                    m[i][j] = service_val[i] * anomaly_val[j]
            product.append(m)
        # 获得每个笛卡尔积矩阵的topk及坐标
        writer = open(self.config['tuple_path'], 'w', encoding='utf8')
        writer.write(
            "case,topk,service,instance,anomaly\n"
        )
        sa_topks = []
        for idx in range(len(product)):
            m = product[idx]
            topk = []
            last_max_val = 1
            for k in range(TOPK_SA):
                cur_max_val = tensor(0)
                x = 0
                y = 0
                for i in range(N_S):
                    for j in range(N_A):
                        if m[i][j] > cur_max_val and m[i][j] < last_max_val:
                            cur_max_val = m[i][j]
                            x = i
                            y = j
                topk.append(((x, y), cur_max_val.item()))
                last_max_val = cur_max_val
            case, num_dict = ins_event_num_list[idx]
            
            for k, _topk in enumerate(topk):
                s_idx, a_idx = _topk[0]

                temp = sorted([(ins_id, num_dict[ins_id]) for ins_id in self.topoinfo[s_idx]],
                              key=lambda x: x[-1], reverse=True)

                row = "%s,%s,%s,%s,%s\n" %(str(case),str(k),self.idx2service[s_idx],self.idx2instance[temp[0][0]],self.idx2anomaly[a_idx])
                writer.write(row)
            sa_topks.append(topk)
        writer.close()
        return sa_topks

    def do_lab(self, action, test_type=None):
        save_dir = self.config['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        RawDataProcess(self.config).process()
        # 训练
        if action == 'train':
            t1 = time.time()
            print('train starts at', t1)
            model_ts, model_ta = self.multi_trainv0(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
                                                                  os.path.join(save_dir, 'train_ys_service.pkl'),
                                                                  os.path.join(save_dir, 'topology.pkl'),
                                                                  aug=self.config['aug'],
                                                                  aug_size=self.config['aug_size'],
                                                                  shuffle=True),
                                                    UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
                                                                  os.path.join(save_dir, 'train_ys_failure_type.pkl'),
                                                                  os.path.join(save_dir, 'topology.pkl'),
                                                                  aug=self.config['aug'],
                                                                  aug_size=self.config['aug_size'],
                                                                  shuffle=True))

            t2 = time.time()
            print('train ends at', t2)
            print('train use time', t2 - t1, 's')
            # 保存模型
            torch.save(model_ts, self.config['service_model_path'])
            torch.save(model_ta, self.config['failure_type_model_path'])
        elif action == 'test':
            service_model_path = self.config['service_model_path']
            failure_type_model_path = self.config['failure_type_model_path']
            model_ts = torch.load(service_model_path)
            model_ta = torch.load(failure_type_model_path)
            if test_type == 'instance':
                print('instance')
                s_outputs, s_labels = self.testv2(model_ts,
                                UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
                                            os.path.join(save_dir, 'train_ys_service.pkl'),
                                            os.path.join(save_dir, 'topology.pkl')),
                                'instance',
                                'instance_pred_multi_v0.csv',
                                'instance_acc_multi_v0.csv')
                out_dir = os.path.join(self.config['save_dir'], 'preds', 'instance_pred_multi_v0.csv')
                ins_res = pd.read_csv(out_dir, index_col=0)
                ins_res = ins_res.to_json(orient='records')
                return ins_res
            elif test_type == 'anomaly':
                print('anomaly type')
                a_outputs, a_labels = self.testv2(model_ta,
                                UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
                                            os.path.join(save_dir, 'train_ys_failure_type.pkl'),
                                            os.path.join(save_dir, 'topology.pkl')),
                                'failure_type',
                                'failure_pred_multi_v0.csv',
                                'failure_acc_multi_v0.csv')
                a_values = nn.Softmax(dim=1)(a_outputs)
                a_values = a_values.tolist()
                a_labels = a_labels.tolist()
                return {
                    'probabilities': a_values,
                    'y_true': a_labels
                }