# Getting Started

## 接口

**model_train**: 

输入：start_time, end_time

输出：message: 'SUCCESS'

描述：此接口只是为了测试进行模型训练保留

**model_test**:

输入：start_time, end_time

输出：  
```
        {
            "fault_id": 0,
            "start_time": "2023-06-09 09:35:13",
            "end_time": "2023-06-09 09:36:13",
            "pred": "k8s容器网络资源包重复发送", //预测
            "groundtruth": "k8s容器进程中止" //真实故障
        }
```
描述：使用训练好的模型，对测试集进行检测

**get_failure_info**:

输入：fault_id

输出：  
```
        {
            "pod": "checkoutservice",    // 故障发生的pod
            "failure_type_pred": "k8s容器网络资源包重复发送",   //预测的故障类型
            "failure_type_ground_truth": "k8s容器进程中止",   //真实的故障类型
            "instance_ground_truth": "checkoutservice",      //真实的故障实例 == pod
            "instance_pred": [ //topk 预测的故障实例
                {
                    "name": "top1",
                    "value": "redis"
                },
                {
                    "name": "top2",
                    "value": "recommendationservice"
                },
                {
                    "name": "top3",
                    "value": "cartservice"
                },
                {
                    "name": "top4",
                    "value": "productcatalogservice"
                },
                {
                    "name": "top5",
                    "value": "checkoutservice"
                }
            ],
            "table": [ 
                {
                    "name": "真实故障",
                    "score": 100,
                    "instance": "checkoutservice",
                    "type": "k8s容器进程中止"
                },
                {
                    "name": "故障1",
                    "score": 100,
                    "instance": "redis",
                    "type": "k8s容器网络资源包重复发送"
                },
                {
                    "name": "故障2",
                    "score": 100,
                    "instance": "recommendationservice",
                    "type": "k8s容器网络资源包重复发送"
                },
                {
                    "name": "故障3",
                    "score": 100,
                    "instance": "cartservice",
                    "type": "k8s容器网络资源包重复发送"
                },
                {
                    "name": "故障4",
                    "score": 100,
                    "instance": "productcatalogservice",
                    "type": "k8s容器网络资源包重复发送"
                },
                {
                    "name": "故障5",
                    "score": 100,
                    "instance": "checkoutservice",
                    "type": "k8s容器网络资源包重复发送"
                }
            ]
        }
```
描述：返回详细的预测结果


## Parameter Description in the Demo
### fastText \& Instance Embedding
* `vector_dim`: The dimension of event embedding vectors. (default: 100)
* `sample_count`: The number of samples per type after data augmentation. (default: 1000)
* `edit_count`: The number of events modified per sample during data augmentation. (default: 1)
* `minCount`: The minimum number of occurrences of the event (events that occur less than this number are ignored). (default: 1)
### DGL
* `epoch`: Training rounds. (default: 6000)
* `batch_size`: The number of samples contained in a batch of data. (default: 1000)
* `win_size`: The length of the judgment window for ending training early. (default: 10)
* `win_threshole`: The thresh for ending training early. (default: 0.0001)
* `lr`: The learning rate. (default: 0.001)

