import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
import csv
from datetime import datetime, timedelta
import time
import json
import itertools
from itertools import combinations, chain
from scipy.stats import norm, pearsonr
import math
import networkx as nx
import random
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class AutoMap:
    def __init__(self, config):
        self.config = config

    def process_data(self, origin_metric_dir):
        run_table = pd.read_csv(self.config['run_table_path'])

        metric_dir = os.path.join(origin_metric_dir, 'container')
        file_names = os.listdir(metric_dir)
        file_names = [file_name for file_name in file_names if file_name.endswith('.csv')]

        data_list = []
        for file_name in file_names:
            data = pd.read_csv(metric_dir + '/' + file_name)
            data_list.append(data)
        all_data = pd.concat(data_list)
        all_data = all_data.reset_index(drop=True)
        print('-'*10 + '指标合并完成' + '-'*10)

        # 按“节点-指标”格式分类指标
        output_dir = self.config['metric_dir']
        have_table_header = set()
        length = len(all_data)
        with tqdm(total=length) as pbar:
            for index, row in all_data.iterrows():
                instance_name = row['cmdb_id'].split('.')[1]
                kpi_name = row['kpi_name'].split('.')[0]
                name_csv = instance_name + '_' + kpi_name

                if name_csv not in have_table_header:
                    with open(output_dir + '/' + name_csv + '.csv', 'a') as f:
                        header = "timestamp,value\n"
                        f.write(header)
                        f.flush()
                        f.close()
                    have_table_header.add(name_csv)

                with open(output_dir + '/' + name_csv + '.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([row['timestamp'], row['value']])
                    f.flush()
                    f.close()

                pbar.update(1)
            print(11)
        print(12)
        metric_path = self.config['metric_dir']
        types = self.config['types'].split()
        instances = self.config['instances'].split()
        sequence_num = self.config['sequence_num']

        # 按照case划分
        all_list = []
        for i in tqdm(range(len(run_table))):
            st_time = run_table['start_time'][i]
            st_array = datetime.strptime(st_time, "%Y-%m-%d %H:%M:%S")
            st_stamp = int(time.mktime(st_array.timetuple()))
            cases_list = []
            # 按照type划分
            for j in range(len(types)):
                types_list = []
                # 按照实例划分
                for k in range(len(instances)):
                    temp_df = pd.read_csv(metric_path+'/'+instances[k]+'_'+types[j]+'.csv')
                    temp_df = temp_df.loc[temp_df['timestamp'].astype(int) >= st_stamp]
                    # 取sequence_num个值
                    temp_list = list(temp_df['value'])[:sequence_num]
                    types_list.append(np.array(temp_list))
                cases_list.append(np.array(types_list))
            all_list.append(np.array(cases_list))

        processed_path = self.config['processed_data_dir']
        if os.path.exists(processed_path):
            shutil.rmtree(processed_path)
        os.makedirs(processed_path)
        # 每个case保存一个预处理文件
        for i in range(len(all_list)):
            np.save(processed_path + '/' + str(i) + '.npy', all_list[i])

    def process_groundtruth(self, origin_groundtruth_dir):
        # 将folder_path下的所有ground_truth文件重构成dataframe，并且拼凑成一个完整的dt
        dt = pd.DataFrame()
        for file_path in os.listdir(origin_groundtruth_dir):
            with open(origin_groundtruth_dir + '/' + file_path, 'r', encoding='utf-8') as file:
                json_content = file.read()
            parsed_json = json.loads(json_content)
            start_time_list = [datetime.fromtimestamp(ts) for ts in parsed_json['timestamp']]
            end_time_list = [start_time + timedelta(minutes=10) for start_time in start_time_list]
            print(3)
            temp_dt = pd.DataFrame({
                'timestamp': parsed_json['timestamp'],
                'start_time': start_time_list,
                'end_time': end_time_list,
                'cmdb_id': parsed_json['cmdb_id'],
                'failure_type': parsed_json['failure_type']
            })
            print(4)
            dt = pd.concat([dt, temp_dt], ignore_index=True)
        print(5)
        dt = dt.sort_values(by='timestamp').reset_index(drop=True)
        print(6)
        save_path = self.config['run_table_path']
        if os.path.exists(save_path):
            os.remove(save_path)
        dt.index.name = 'case_id'
        dt.to_csv(save_path)

    # PC实现
    def subset(self, iterable):
        """
        求子集: subset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        """
        xs = list(iterable)
        # 返回 iterator 而不是 list
        return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))

    def skeleton(self, suffStat, indepTest, alpha, labels, m_max):
        sepset = [[[] for i in range(len(labels))] for i in range(len(labels))]

        # 完全无向图
        G = [[True for i in range(len(labels))] for i in range(len(labels))]
        W = [[[1 for i in range(4)] for i in range(len(labels))] for i in range(len(labels))]
        for i in range(len(labels)):
            G[i][i] = False  # 不需要检验 i -- i
            W[i][i] = [0, 0, 0, 0]
        done = False  # done flag

        ord = 0
        n_edgetests = {0: 0}
        while done != True and any(G) and ord <= m_max:
            ord1 = ord + 1
            n_edgetests[ord1] = 0

            done = True

            # 相邻点对
            ind = []
            for i in range(len(G)):
                for j in range(len(G[i])):
                    if G[i][j] == True:
                        ind.append((i, j))

            G1 = G.copy()

            for x, y in ind:
                if G[x][y] == True:
                    neighborsBool = [row[x] for row in G1]
                    neighborsBool[y] = False

                    # adj(C,x) \ {y}
                    neighbors = [i for i in range(len(neighborsBool)) if neighborsBool[i] == True]

                    if len(neighbors) >= ord:

                        # |adj(C, x) \ {y}| > ord
                        if len(neighbors) > ord:
                            done = False

                        # |adj(C, x) \ {y}| = ord
                        for neighbors_S in set(itertools.combinations(neighbors, ord)):
                            n_edgetests[ord1] = n_edgetests[ord1] + 1

                            # 节点 x, y 是否被 neighbors_S d-seperation
                            # 条件独立性检验，返回 p-value
                            cpu = pd.DataFrame(suffStat[0])
                            cpu = pd.DataFrame(cpu.values.T)
                            cpu_C, cpu_n = cpu.corr().values, cpu.values.shape[0]
                            diskio = pd.DataFrame(suffStat[1])
                            diskio = pd.DataFrame(diskio.values.T)
                            diskio_C, diskio_n = diskio.corr().values, diskio.values.shape[0]
                            memory = pd.DataFrame(suffStat[2])
                            memory = pd.DataFrame(memory.values.T)
                            memory_C, memory_n = memory.corr().values, memory.values.shape[0]
                            network = pd.DataFrame(suffStat[3])
                            network = pd.DataFrame(network.values.T)
                            network_C, network_n = network.corr().values, network.values.shape[0]

                            cpu_pval = indepTest(cpu_C, cpu_n, x, y, list(neighbors_S))

                            diskio_pval = indepTest(diskio_C, diskio_n, x, y, list(neighbors_S))

                            memory_pval = indepTest(memory_C, memory_n, x, y, list(neighbors_S))

                            network_pval = indepTest(network_C, network_n, x, y, list(neighbors_S))

                            if cpu_pval >= alpha:
                                W[x][y][0] = 0
                            if diskio_pval >= alpha:
                                W[x][y][1] = 0
                            if memory_pval >= alpha:
                                W[x][y][2] = 0
                            if network_pval >= alpha:
                                W[x][y][3] = 0
                            if sum(W[x][y]) != 0:
                                W[x][y] = list(np.array(W[x][y])/sum(W[x][y]))
                            # 条件独立
                            if cpu_pval >= alpha or diskio_pval >= alpha or memory_pval >= alpha or network_pval >= alpha:
                                G[x][y] = False

                                # 把 neighbors_S 加入分离集
                                sepset[x][y] = list(neighbors_S)
                                break

            ord += 1

        return {'sk': np.array(G), 'sepset': sepset}, W

    def extend_cpdag(self, graph):
        def rule1(pdag, solve_conf=False, unfVect=None):
            """Rule 1: 如果存在链 a -> b - c，且 a, c 不相邻，把 b - c 变为 b -> c"""
            search_pdag = pdag.copy()
            ind = []
            for i in range(len(pdag)):
                for j in range(len(pdag)):
                    if pdag[i][j] == 1 and pdag[j][i] == 0:
                        ind.append((i, j))

            #
            for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
                isC = []

                for i in range(len(search_pdag)):
                    if (search_pdag[b][i] == 1 and search_pdag[i][b] == 1) and (search_pdag[a][i] == 0 and search_pdag[i][a] == 0):
                        isC.append(i)

                if len(isC) > 0:
                    for c in isC:
                        if 'unfTriples' in graph.keys() and ((a, b, c) in graph['unfTriples'] or (c, b, a) in graph['unfTriples']):
                            # if unfaithful, skip
                            continue
                        if pdag[b][c] == 1 and pdag[c][b] == 1:
                            pdag[b][c] = 1
                            pdag[c][b] = 0
                        elif pdag[b][c] == 0 and pdag[c][b] == 1:
                            pdag[b][c] = pdag[c][b] = 2

            return pdag

        def rule2(pdag, solve_conf=False):
            """Rule 2: 如果存在链 a -> c -> b，把 a - b 变为 a -> b"""
            search_pdag = pdag.copy()
            ind = []

            for i in range(len(pdag)):
                for j in range(len(pdag)):
                    if pdag[i][j] == 1 and pdag[j][i] == 1:
                        ind.append((i, j))

            #
            for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
                isC = []
                for i in range(len(search_pdag)):
                    if (search_pdag[a][i] == 1 and search_pdag[i][a] == 0) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                        isC.append(i)
                if len(isC) > 0:
                    if pdag[a][b] == 1 and pdag[b][a] == 1:
                        pdag[a][b] = 1
                        pdag[b][a] = 0
                    elif pdag[a][b] == 0 and pdag[b][a] == 1:
                        pdag[a][b] = pdag[b][a] = 2

            return pdag

        def rule3(pdag, solve_conf=False, unfVect=None):
            """Rule 3: 如果存在 a - c1 -> b 和 a - c2 -> b，且 c1, c2 不相邻，把 a - b 变为 a -> b"""
            search_pdag = pdag.copy()
            ind = []
            for i in range(len(pdag)):
                for j in range(len(pdag)):
                    if pdag[i][j] == 1 and pdag[j][i] == 1:
                        ind.append((i, j))

            #
            for a, b in sorted(ind, key=lambda x:(x[1], x[0])):
                isC = []

                for i in range(len(search_pdag)):
                    if (search_pdag[a][i] == 1 and search_pdag[i][a] == 1) and (search_pdag[i][b] == 1 and search_pdag[b][i] == 0):
                        isC.append(i)

                if len(isC) >= 2:
                    for c1, c2 in combinations(isC, 2):
                        if search_pdag[c1][c2] == 0 and search_pdag[c2][c1] == 0:
                            # unfaithful
                            if 'unfTriples' in graph.keys() and ((c1, a, c2) in graph['unfTriples'] or (c2, a, c1) in graph['unfTriples']):
                                continue
                            if search_pdag[a][b] == 1 and search_pdag[b][a] == 1:
                                pdag[a][b] = 1
                                pdag[b][a] = 0
                                break
                            elif search_pdag[a][b] == 0 and search_pdag[b][a] == 1:
                                pdag[a][b] = pdag[b][a] = 2
                                break

            return pdag

        # Rule 4: 如果存在链 i - k -> l 和 k -> l -> j，且 k 和 l 不相邻，把 i - j 改为 i -> j
        # 显然，这种情况不可能存在，所以不需要考虑 rule4

        pdag = [[0 if graph['sk'][i][j] == False else 1 for i in range(len(graph['sk']))] for j in range(len(graph['sk']))]

        ind = []
        for i in range(len(pdag)):
            for j in range(len(pdag[i])):
                if pdag[i][j] == 1:
                    ind.append((i, j))

        # 把 x - y - z 变为 x -> y <- z
        for x, y in sorted(ind, key=lambda x:(x[1],x[0])):
            allZ = []
            for z in range(len(pdag)):
                if graph['sk'][y][z] == True and z != x:
                    allZ.append(z)

            for z in allZ:
                if graph['sk'][x][z] == False \
                    and graph['sepset'][x][z] != None \
                    and graph['sepset'][z][x] != None \
                    and not (y in graph['sepset'][x][z] or y in graph['sepset'][z][x]):
                    pdag[x][y] = pdag[z][y] = 1
                    pdag[y][x] = pdag[y][z] = 0

        # 应用 rule1 - rule3
        pdag = rule1(pdag)
        pdag = rule2(pdag)
        pdag = rule3(pdag)

        return np.array(pdag)

    def pc(self, suffStat, alpha, labels, indepTest, m_max=float("inf"), verbose=False):
        # 骨架
        graphDict, Weight = self.skeleton(suffStat, indepTest, alpha, labels, m_max)
        # 扩展为 CPDAG
        cpdag = self.extend_cpdag(graphDict)
        # 输出贝叶斯网络图矩阵
        if verbose:
            # print(cpdag)
            pass
        return cpdag, Weight

    def gauss_ci_test(self, C, n, x, y, S):
        """条件独立性检验"""

        cut_at = 0.9999999

        # ------ 偏相关系数 ------
        # S中没有点
        if len(S) == 0:
            r = C[x, y]

        # S 中只有一个点，即一阶偏相关系数
        elif len(S) == 1:
            r = (C[x, y] - C[x, S] * C[y, S]) / math.sqrt((1 - math.pow(C[y, S], 2)) * (1 - math.pow(C[x, S], 2)))
            

        # 其实我没太明白这里是怎么求的，但 R 语言的 pcalg 包就是这样写的
        else:
            m = C[np.ix_([x]+[y] + S, [x] + [y] + S)]
            m = np.nan_to_num(m)
            PM = np.linalg.pinv(m)

            r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))

        r = min(cut_at, max(-1 * cut_at, r))

        # Fisher’s z-transform
        res = math.sqrt(n - len(S) - 3) * .5 * math.log1p((2 * r) / (1 - r))

        # Φ^{-1}(1-α/2)
        return 2 * (1 - norm.cdf(abs(res)))

    # 随机游走实现
    def remove_self(self, list_a, a):
        list_a = list(list_a)
        if a in list_a:
            list_a.remove(a)
            return list_a
        else:
            return list_a

    def construct_graph(self, V, E):
        """
        networkx构建有向图
        V list 点列表
        E np.array 边的邻接矩阵
        """
        assert len(V) == E.shape[0]
        assert len(V) == E.shape[1]
        G = nx.DiGraph()
        G.add_nodes_from(list(range(len(V))))
        for i in range(len(V)):
            for j in range(len(V)):
                if E[i, j] != 0:
                    G.add_edge(i, j)  # 索引
                    # G.add_edge(V[i], V[j]) #名字
        # # 画图
        # import matplotlib.pyplot as plt
        # pos = nx.circular_layout(G) # 圆形布局 起到美化作用
        # nx.draw(G, pos, with_labels=True, font_weight="bold")
        # plt.title("DiGraph")
        # plt.show()
        return G

    def probablity_matrix(self, G, W, C, v_fe, r=0.2):
        '''
        计算概率矩阵
        G nx.DiGraph() 根因定位图
        W np.array 节点间对应指标是否相关
        C np.array 节点间对应指标相关性
        v_fe int 游走起点
        r int 跳前驱节点系数
        return
        P pd.DataFrame n*n大小 标准化的概率矩阵 由相似性度量仅有P.loc[i, i.succ]非0
        '''
        n = len(G.nodes())  # 节点数
        m = W.shape[-1]  # 指标数
        P = pd.DataFrame(np.zeros((n, n), dtype=np.float64),
                        index=G.nodes(), columns=G.nodes())
        # print(P)
        for i in G.nodes():
            # 处理后继节点
            successor = list(G.succ[i]) if i in G.succ[i] else list(
                G.succ[i]) + [i]
            # print(successor)
            for j in successor:
                for k in range(m):
                    P.loc[i, j] += 1/m * W[i, j, k] * (C[j, v_fe, k]/(sum([C[l, v_fe, k] for l in successor]))) if sum([
                        C[l, v_fe, k] for l in successor]) != 0 else 0
            # 处理前驱节点
            predcessor = self.remove_self(G.pred[i], i)
            for j in predcessor:
                P.loc[i, j] = r * (P.loc[j, i] / sum([P.loc[l, i] for l in predcessor])
                                ) if sum([P.loc[l, i] for l in predcessor]) != 0 else 0
            # 处理自跳
            c_self = P.loc[i, i]
            if c_self > P.loc[i].max():
                P.loc[i, i] = c_self - P.loc[i].max()
            else:
                P.loc[i, i] = 0
            s = P.loc[i].sum()
            if s == 0: #孤立点平均跳
                # print("孤立点")
                for j in G.nodes():
                    P.loc[i, j] = 1 / n
                continue
            for j in set(predcessor+successor):
                P.loc[i, j] = P.loc[i, j] / s
            # print(P.loc[i].sum())
            # input()
        return P

    def random_pick(self, some_list, probabilities):
        '''
        somelist 项列表
        probablities 概率列表
        return
        返回所选的项
        '''
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        for item, item_probability in zip(some_list, probabilities):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                break
        return item

    def random_walk_test(self, G, P, v_fe, num_loop):
        '''
        G nx.DiGraph() 根因定位图
        P pd.DataFrame n*n大小 标准化的概率矩阵 由相似性度量仅有 P.loc[i, i.succ]非0
        v_fe 前端节点
        num_loop 循环次数
        v_s 当前节点
        v_p 前一个节点
        R 字典 以节点名为键 出现次数为值
        return
        返回一个排序后的列表
        '''
        # 初始化
        v_s = v_fe
        v_p = v_fe
        R = dict.fromkeys(G.nodes(), 0)

        # 循环随机游走
        for i in range(num_loop):
            v_p = v_s
            v_s = self.random_pick(P.columns.tolist(), P.loc[v_s].values)
            R[v_s] += 1
        #         print(f"loop {i}: {v_p} -> {v_s}")

        R_order = sorted(R.items(), key=lambda x: x[1], reverse=True)
        return R_order

    def train(self, node_to_drop=None, num_loop=500):
        v = self.config['instances'].split()
        result_df_list = []
        spend_time_list = []
        folder_path = self.config['processed_data_dir']
        result_path = self.config['E_W_C_dir']
        files = os.listdir(folder_path)
        file_num = len(files)
        case_to_ignore = []
        # 循环待定，依据数据集调整
        for gaia in tqdm(range(file_num)):
            case = np.load(folder_path + '/' + str(gaia)+'.npy')
            start = time.perf_counter()
            try:
                p, w = self.pc(
                    suffStat = case,
                    alpha = 0.5,
                    labels = v,
                    indepTest = self.gauss_ci_test,
                    verbose = True
                )
            except Exception as e:
                print(e)
                case_to_ignore.append(gaia)
                continue

            c = []
            for i in range(len(v)):
                c_i = []
                for j in range(len(v)):
                    c_i_j = []
                    for k in range(4):
                        case_i_k = case[k][i]
                        case_j_k = case[k][j]
                        covariance = np.cov(case_i_k,case_j_k)
                        if (np.std(case_i_k) * np.std(case_j_k)) == 0:
                            covar = covariance
                        else:
                            covar = covariance / (np.std(case_i_k) * np.std(case_j_k))
                        c_i_j_k = np.linalg.det(covar)
                        c_i_j .append(c_i_j_k)
                    c_i.append(c_i_j)
                c.append(c_i)
            os.makedirs(os.path.join(result_path, str(gaia)), exist_ok=True)
            np.save(result_path+'/'+str(gaia)+'/E.npy',p)
            np.save(result_path+'/'+str(gaia)+'/W.npy',w)
            np.save(result_path+'/'+str(gaia)+'/C.npy',c)
            # =====随机游走=====
            E = np.array(p)
            W = np.array(w)
            C = np.array(c)
            # 随机去除某些指标
            v_fe = random.choice(node_to_drop)
            v_fe = v.index(v_fe)
            G = self.construct_graph(v, E)
            P = self.probablity_matrix(G, W, C, v_fe)
            walk_order = self.random_walk_test(G, P, v_fe, num_loop=num_loop)
            elapsed = (time.perf_counter() - start)
            spend_time_list.append(elapsed)
            top5_result = {"index": [gaia],
                        "top1": [v[walk_order[0][0]]],
                        "top2": [v[walk_order[1][0]]],
                        "top3": [v[walk_order[2][0]]],
                        "top4": [v[walk_order[3][0]]],
                        "top5": [v[walk_order[4][0]]],
                        }
            result_df_list.append(pd.DataFrame(top5_result))
        result_df = pd.concat(result_df_list, axis=0)
        result_df['test_time'] = spend_time_list
        return result_df, case_to_ignore

    def run(self, origin_metric_dir, origin_groundtruth_dir):
        # 数据预处理
        print('data preprocessing')
        self.process_groundtruth(origin_groundtruth_dir)
        self.process_data(origin_metric_dir)

        # 开始训练
        print('start training')
        epoch = self.config['epoch']
        results = []
        node_to_drop = self.config['node_to_drop'].split()
        num_loop = self.config['num_loop']
        for i in range(epoch):
            run_table = pd.read_csv(self.config['run_table_path'])
            temp_df = pd.DataFrame()
            result_df, case_to_drop = self.train(node_to_drop=node_to_drop, num_loop=num_loop)

            run_table.drop(index=case_to_drop, inplace=True)
            
            temp_df["top1"] = result_df["top1"]
            temp_df["top2"] = result_df["top2"]
            temp_df["top3"] = result_df["top3"]
            temp_df["top4"] = result_df["top4"]
            temp_df["top5"] = result_df["top5"]
            temp_df["groundtruth"] = run_table["cmdb_id"]

            results.append(temp_df.to_json(orient='records'))
        # 转换为字典格式
        result = results[-1]
        return result
