# AutoMAP算法说明文档

## 项目结构

automap

--metric：原始数据，文件格式为“instance_指标.csv”

--processed：预处理完成后的数据，shape为“指标x实例x时间窗口”，如“4x10x10”，表示4个指标，10个实例，选取10个点（每个点为1min）

--test_time_E_W_C

--image：训练结果图

--automap.py

--config.yaml

--public_function.py

--result.json：实验结果

--run_table.csv：由groundtruth得到的csv

--README.md

--requirements.txt

### automap.py

- 类AutoMap

```
方法subset, skeleton, extend_cpdag, pc, gauss_ci_test是关于PC实现的内容
方法remove_self, construct_graph, probability_matrix, random_pick, random_walk_test是关于随机游走实现的内容
方法process_data是处理指标数据
方法process_groundtruth是处理groundtruth文件
方法train是automap算法的训练方法
方法run是运行automap方法，以及保存实验结果
```

### config.yaml

```
以数据集为键编写配置文件
data_folder_list是指标文件夹路径
ground_truth_path是groundtruth文件夹路径
types是automap选择训练的指标
instances是数据集的实例
sequence_num是选择指标的窗口数量，一个窗口1min
node_to_drop是automap需要忽略的结点，随机选取
result_path是结果文件保存的路径
```

```
如果希望多选择几个指标进行训练，则需要修改automap.py里的skeleton方法和config.yaml里的types字段
```

## 环境配置

python=3.9.18

```
pip install -r requirements.txt
```

## 执行方法

1. 修改 main 里 `AutoMap('./config.yaml', 'aiops')` 的 `aiops`字段，对应的是使用配置文件里哪个数据集

2. 修改配置文件内容：`data_folder_list`, `ground_truth_path`, `types`, `instances`, `node_to_drop`

3. 运行 `python automap.py`