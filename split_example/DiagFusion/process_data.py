import pandas as pd
from tqdm import tqdm
import os
import csv
import numpy as np
import pickle
import json
from datetime import *
from Drain3.drain3.template_miner import TemplateMiner
from Drain3.drain3.template_miner_config import TemplateMinerConfig
from sklearn.model_selection import train_test_split
import sys
import pathlib
DF_path = pathlib.Path(__file__).parent
sys.path.append(str(DF_path))
root_path = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_path))
import logging
from detector.k_sigma import Ksigma
import multiprocessing
from copy import copy
from DiagFusion.public_function import save
import math

def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
        
def create_run_table(processed_data_config):
    # 将groundtruth转换为run_table
    data_path = os.path.join(processed_data_config['groundtruth_path'], 'groundtruth.json')
    with open(data_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)
    start_time = [ts for ts in json_data['timestamp']]
    end_time = [st + duration for st, duration in zip(start_time, json_data['duration'])]
    df = pd.DataFrame({
        'service': json_data['service'],
        'instance': json_data['cmdb_id'],
        'failure_type': json_data['failure_type'],
        'start_time': start_time,
        'end_time': end_time
    })
    df.to_csv(processed_data_config['run_table_path'], index=False)


def generate_key(instance, failure_instance, failure_type):
    if instance == failure_instance:
        return instance, '[' + failure_type + ']'
    else:
        return instance, '[normal]'


def process_task(df, case_id, st_time, ed_time):
    detector = Ksigma()
    rt = []
    scheduler = tqdm(total=len(df), desc=f"case:{case_id}, detecting")
    for instance, ins_group in df.groupby(by="cmdb_id"):
        for kpi, kpi_group in ins_group.groupby(by="kpi_name"):
            res = detector.detection(kpi_group, "value", st_time, ed_time)
            if res[0] is True:
                rt.append([int(res[1]), instance, kpi, res[2]])
        scheduler.update(len(ins_group))
    return rt


def process_metric(processed_data_config):
    run_table = pd.read_csv(processed_data_config['run_table_path'])
    
    print("处理metric")
    import os

    metric_file_path_list = []
    for dirpath, _, filenames in os.walk(processed_data_config["origin_metric_dir"]):
        for filename in filenames:
            if filename.find(".csv") != -1:
                metric_file_path_list.append(os.path.join(dirpath, filename))

    metric_df = None
    for filepath in tqdm(metric_file_path_list, desc="加载metric数据"):
        data = pd.read_csv(filepath)
        cmdb_id = filepath.split("_")[0]
        kpi_name = "_".join(filepath.split("_")[1:])
        data["cmdb_id"] = [cmdb_id] * len(data)
        data["kpi_name"] = [kpi_name] * len(data)
        if metric_df is None:
            metric_df = data
        else:
            metric_df = pd.concat([metric_df, data])

    metric_dict = {}
    tasks = []
    pool = multiprocessing.Pool(processes=10)
    for case_id, case in run_table.iterrows():
        # 故障前60个点，故障后0个点
        sample_interval = 60
        st_time = case["start_time"] - (sample_interval * 60)
        ed_time = case["end_time"] + (sample_interval * 0)
        task = pool.apply_async(
            process_task,
            (
                metric_df.query(f"timestamp >= {st_time} & timestamp < {ed_time}"),
                case_id,
                st_time,
                ed_time,
            ),
        )
        tasks.append((case_id, task))
        # 每个实例，每个指标采样
    pool.close()
    pool.join()
    for case_id, task in tasks:
        metric_dict[case_id] = task.get()

    os.makedirs(processed_data_config["metric_dir"], exist_ok=True)
    with open(processed_data_config["metric_path"], "w") as w:
        json.dump(metric_dict, w)

    print("处理完成")

def hash_decimal_to_hex(decimal):
    # 使用Python内置哈希函数将整数哈希为一个大整数
    hashed_decimal = abs(hash(str(decimal)))
    # 将大整数转换为16进制字符串
    hex_string = hex(hashed_decimal)
    # 取字符串末尾8个字符作为哈希值，即一个长度为8的16进制数
    hash_value = hex_string[:8]
    # 将16进制数转换为整数并返回
    return hash_value


def get_drain_template(log_df):
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    config = TemplateMinerConfig()
    config.load("Drain3/drain3.ini")
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    lines = log_df["message"].tolist()

    result_json_list = []
    for line in tqdm(lines, desc="draining"):
        line = str(line).rstrip()

        result = template_miner.add_log_message(line)

        result_json_list.append(result)

    sorted_clusters = sorted(
        template_miner.drain.clusters, key=lambda it: it.size, reverse=True
    )
    for cluster in sorted_clusters:
        logger.info(cluster)

    EID_list = []
    for logdict in result_json_list:
        EID_list.append(hash_decimal_to_hex(logdict["cluster_id"]))

    log_df["EventId"] = EID_list

    return log_df


def stratified_sampling(df: pd.DataFrame, run_table, save_path):
    logs_list = []
    for i in tqdm(range(0, len(run_table)), desc="日志采样："):
        service_list = []
        temp_df = df.loc[
            (df["timestamp"] >= run_table["start_time"][i])
            & (df["timestamp"] <= run_table["end_time"][i])
        ]
        unique_list = np.unique(temp_df["EventId"], return_counts=True)
        event_id = unique_list[0]
        cnt = unique_list[1]
        for k in range(len(cnt)):
            if cnt[k] == 1:
                unique_log = (
                    temp_df[temp_df["EventId"] == event_id[k]].T.to_dict().values()
                )
                unique_log = list(unique_log)[0]
                service_list.append(
                    [
                        unique_log["timestamp"],
                        unique_log["cmdb_id"],
                        unique_log["EventId"],
                    ]
                )
                temp_df = temp_df[temp_df["EventId"] != event_id[k]]
        X = temp_df
        y = temp_df["EventId"]

        class_num = len(event_id)

        if len(temp_df) == 0:
            logs_list.append([])
            continue
        elif len(temp_df) < class_num and len(temp_df) >= 1:
            X_test = temp_df
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=class_num, stratify=y
            )
        for _, row in X_test.iterrows():
            service_list.append([row["timestamp"], row["cmdb_id"], row["EventId"]])
        logs_list.append(service_list)
    logs_list = np.array(logs_list, dtype=object)
    np.save(save_path, logs_list)


def process_log(processed_data_config):
    run_table = pd.read_csv(processed_data_config['run_table_path'])
    print("处理log")
    print("读取日志")
    import os

    log_df = None
    log_file_path_list = []
    for dirpath, _, filenames in os.walk(processed_data_config["origin_log_dir"]):
        for filename in filenames:
            if filename.find("log") != -1:
                full_log_path = os.path.join(dirpath, filename)
                log_file_path_list.append(full_log_path)
    for path in log_file_path_list:
        log_data = pd.read_csv(path)
        if log_data is None:
            log_df = log_data
        else:
            log_df = pd.concat([log_df, log_data])
        print(f"成功处理{path}")

    
    os.makedirs(processed_data_config['log_dir'], exist_ok=True)
    log_df["timestamp"] = log_df["timestamp"].apply(lambda x: int(x))
    print("提取模板，准备drain")
    log_template_df = get_drain_template(log_df)
    print("准备采样")
    stratified_sampling(log_template_df, run_table, processed_data_config["log_path"])
    print("处理完成")
    

def sub_task(run_table_path, trace_file_path, index):
    try:
        run_table = pd.read_csv(run_table_path)
        trace_df = pd.read_csv(trace_file_path)
        detector = Ksigma()
        # 转换时间戳
        trace_df["timestamp"] = trace_df["timestamp"].apply(lambda x: int(x / 1000000))

        trace_df = trace_df.rename(columns={"parent_span": "parent_id"})
        # 父子拼接
        meta_df = trace_df[["parent_id", "cmdb_id"]].rename(
            columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
        )

        trace_df = pd.merge(trace_df, meta_df, on="span_id")

        # 按事件排序
        trace_df = trace_df.sort_values(by="timestamp")
        time_series = trace_df["timestamp"].values.tolist()

        st_time = copy(time_series[0])
        ed_time = copy(time_series[-1])

        sche = tqdm(total=len(trace_df), desc=f"调用链收集{trace_file_path}")

        # 每个 60s | 1min 统计一次
        interval = 1
        trace_dict = {}
        for case_id, case in run_table.iterrows():
            trace_dict[case_id] = []

        sample_cnt = int((ed_time - st_time) / interval) + 1
        interval_info = {
            "caller": [],
            "callee": [],
            "timestamp": [],
            "ec":[],
            "lagency": [],
        }

        for caller, caller_group in trace_df.groupby(by="cmdb_id"):
            for callee, callee_group in caller_group.groupby(by="ccmdb_id"):
                for stop_point in range(sample_cnt):
                    sample_time = st_time + stop_point * interval
                    chosen = callee_group[
                        (callee_group["timestamp"] >= sample_time)
                        & (callee_group["timestamp"] < sample_time + interval)
                    ]
                    if len(chosen) == 0:
                        continue
                    cur_lagency = max(0, np.mean(chosen["duration"].values.tolist()))
                    cur_ec = len(chosen.query("status_code not in ['0','200','Ok','OK',0,200]"))
                    interval_info["caller"].append(caller)
                    interval_info["callee"].append(callee)
                    interval_info["ec"].append(cur_ec)
                    interval_info["lagency"].append(cur_lagency)
                    interval_info["timestamp"].append(sample_time)
                sche.update(len(callee_group))
        sche = tqdm(total=len(run_table), desc=f"调用链处理{trace_file_path}")
        interval_info = pd.DataFrame(interval_info)
        for case_id, case in run_table.iterrows():
            # 故障前60分钟至故障结束后0分钟
            cst_time = case["start_time"] - 60 * 60
            ced_time = case["end_time"] + 60 * 0
            case_df = interval_info[
                (interval_info["timestamp"] >= cst_time)
                & (interval_info["timestamp"] < ced_time)
            ]
            for caller, caller_group in case_df.groupby(by="caller"):
                for callee, callee_group in caller_group.groupby(by="callee"):
                    res1 = detector.detection(
                        callee_group, "lagency", cst_time, ced_time
                    )
                    res2 = detector.detection(callee_group, "ec", cst_time, ced_time)
                    if not (res1[0] or res2[0]):
                        continue
                    ts = None
                    if res1[0]:
                        ts = res1[1]
                        score = res1[2]
                    if res2[0]:
                        if ts is None:
                            ts = res2[1]
                        else:
                            ts = min(ts, res2[1])
                        if ts == res2[1]:
                            score = res2[2]
                    trace_dict[case_id].append((int(ts), caller, callee, score))
            sche.update(1)
        return trace_dict
    except Exception as e:
        print(e)


def process_trace(processed_data_config):
    run_table_path = processed_data_config["run_table_path"]
    print("处理trace")

    trace_file_path_list = []
    for dirpath, _, filenames in os.walk(processed_data_config["origin_trace_dir"]):
        for filename in filenames:
            if filename.find("trace") != -1:
                trace_file_path_list.append(os.path.join(dirpath, filename))

    pool = multiprocessing.Pool(processes=max(5, len(trace_file_path_list)))

    tks = []
    for index, filepath in enumerate(trace_file_path_list):
        tk = pool.apply_async(sub_task, (run_table_path, filepath, index))
        tks.append(tk)

    pool.close()
    pool.join()
    trace_dict = None
    for tk in tks:
        data = tk.get()
        if trace_dict is None:
            trace_dict = data
        else:
            for key in trace_dict.keys():
                trace_dict[key].extend(data[key])
    os.makedirs(processed_data_config["trace_dir"], exist_ok=True)
    with open(processed_data_config["trace_path"], "w") as w:
        json.dump(trace_dict, w)
    print("处理完成")

def metric_trace_log_parse(trace, metric, logs, labels, save_path, nodes):
    if not metric is None: # 去除np.inf数值的指标
        for k, v in metric.items():
            metric[k] = [x for x in v if not math.isinf(x[3])]

    if not logs is None:
        logs = list(logs)
        log = {x: [] for x in labels.index}
        if labels.index[-1]+1 == len(log):
            for k, v in log.items():
                log[k] = logs[int(k)]
        else:
            count = 0
            for k, v in log.items():
                log[k] = logs[count]
                count += 1

    service_name = nodes.split()
    anomaly_service = list(labels['instance'])
    anomaly_type = list(labels['failure_type'])

    demo_metric = {x: {} for x in labels.index}
    k = 0
    for case_id, v in tqdm(demo_metric.items()):
        anomaly_service_name = anomaly_service[k]
        anomaly_service_type = anomaly_type[k]
        k += 1
        inner_dict_key = [(x, anomaly_service_type) if x == anomaly_service_name else (x, "[normal]") for x in
                          service_name]
        # 指标
        if not metric is None:
            demo_metric[case_id] = {x: [[y[0], "{}_{}_{}".format(y[1], y[2], "+" if y[3] > 0 else "-")] for y in metric[str(case_id)] if
                                  y[1].find(x[0]) != -1] for x in inner_dict_key}
        else:
            demo_metric[case_id] = {x : [] for x in inner_dict_key}
        # 调用链
        if not trace is None:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend(
                    [[y[0], "{}_{}".format(y[1], y[2])] for y in trace[str(case_id)] 
                     if y[1] == inner_key[0] or y[2] == inner_key[0]])
        # 日志
        if not logs is None:
            for inner_key in inner_dict_key:
                demo_metric[case_id][inner_key].extend([[y[0], y[2]] for y in log[case_id] if y[1] == inner_key[0]])
        for inner_key in inner_dict_key:
            temp = demo_metric[case_id][inner_key]
            sort_list = sorted(temp, key=lambda x: x[0])
            temp_list = [x[1] for x in sort_list]
            demo_metric[case_id][inner_key] = ' '.join(temp_list)

    save(save_path, demo_metric)


def process(processed_data_config):
    trace = None
    metric = None
    logs = None
    run_table = pd.read_csv(processed_data_config['run_table_path'])
    if processed_data_config['log_path']:
        logs = np.load(processed_data_config['log_path'], allow_pickle=True)
    if processed_data_config['metric_path']:
        with open(processed_data_config['metric_path'], 'r', encoding='utf8') as fp:
            metric = json.load(fp)
    if processed_data_config['trace_path']:
        with open(processed_data_config['trace_path'], 'r', encoding='utf8') as fp:
            trace = json.load(fp)
    metric_trace_log_parse(trace, metric, logs, run_table, processed_data_config['save_path'], processed_data_config['nodes'])
