# MicroServo SDK User Guide

## Introduction

👏 Welcome to the MicroServo SDK, a platform's Algorithm Hot-plugging feature designed for deployment model. This platform is dedicated to model deployment services. With its help, users can easily train models and test it. During operation, the platform automatically creates a Docker container to run the algorithm. If you choose to deploy the algorithm online, you can use the `train` method to train your model and then use the `test` method to test it. Alternatively, if you just want to observe the effects of the model, you can choose to use the `run` method to train, test, and obtain results in one go. The path to the original data is written in the system configuration file, which can be directly used in the model methods.


## Getting Started

### Cloning the repository
First, clone our repository by running the following command:

```bash
git clone https://github.com/MicroServo/hot-plugging.git
```

### Adding your algorithm code
After cloning the repository, place your own algorithm in the algorithm folder.

### Reformat your algorithm code
Before you start implementing the training and testing methods for your algorithm, you need to make some modifications to your code to meet our requirements. Initially, all file path information must be integrated into a configuration file at the algorithm directory, using absolute paths. Adherence to naming conventions is mandatory; directories should end with 'dir' and files with 'path'. Subsequently, users must prepare a requirements.txt file detailing all necessary dependencies for the algorithm’s execution.


### Extending the `Algorithm` Class
After you have adjusted the format, you can start to extend the `Algorithm` Class. This class is located in `algorithmTemplate/algorithm_app/algorithm.py`. If you simply want to view the effects of the model without saving it, you only need to implement the `__init__` and `run` methods. 

If you want to deploy the algorithm online to the platform, you will need to implement the __init__, train, and test methods. Additionally, you will need to include the paths of the models to be saved and the paths of intermediate results in the ignore_list within config.yaml located in the root directory.

Before starting, we need to import our own algorithm methods into algorithm.py. We recommend that you encapsulate your algorithm into a class similar to our `Algorithm` class.

### Implementing the `__init__` Method
The `__init__` method is responsible for handling the path information of the algorithm, initializing your model, and providing some variables for the original data. After processing by the `__init__` method, all paths will be modified to start with the `algorithmTemplate/algorithm_app/experiment` prefix, facilitating subsequent management.

### Implementing the `train` Method
The `train` method is used to train your model. MicroServo will provide the original data (with labels) mentioned in the `__init__` method. Here, you need to implement your data preprocessing logic, then use the preprocessed data to train the model, and save the model in the experiment path after training is complete.

### Implementing the `test` Method
The `test` method is used to test the model. MicroServo will read the model trained by the `train` method and the test data (with labels). In this method, you need to return **Json data** in a specific format. MicroServo will store the data in the database and display it on the platform's evaluation data page.

### Implementing the `run` Method
The `run` method does not differentiate between training and testing. MicroServo provides the original data (with labels). In this method, you need to implement data preprocessing, model training, model testing, and then return **Json data** in a specific format. Finally, MicroServo will store the data in the database.

## Example
Here's an example of how to create a custom algorithm using MicroServo SDK:

```Python
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

config_file = 'automap/config.yaml'
class Algorithm:
    def __init__(self, user):
        config = get_config(config_file)
        config = deal_config(config, user, base_dir)
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
            """
            result_dict = {
                'top@k': {
                    0: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1},
                    1: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1},
                    2: {'top1': 0.1, 'top2': 0.1, 'top3': 0.1, 'top4': 0.1, 'top5': 0.1}
                },
                'epoch': 3
            }
            """
            
            result_dict = self.model.run(self.origin_metric_dir, self.origin_groundtruth_dir)
            return result_dict
        except Exception as e:
            print(e)
        finally:
            self.clear()
    
    def clear(self):
        if os.path.exists(self.experiment_dir):
            shutil.rmtree(self.experiment_dir)
```
Feel free to adapt this example to your specific needs and customize your algorithms for MicroServo platform. 🎉🎉

## Debug
After you have completed all operations, don't rush to upload; you can first test it locally. 😎

### Implement the `test` API Interface
MicroServo has reserved a `test` interface in `algorithmTemplate/algorithm_app/views.py`, which is used to test for potential issues such as path problems in each method. In this method, you need to call the method of the `Algorithm` class that you want to test, such as `train`, `test`.

### Copy the data to experiment
We simulate the process of the platform sending data by copying data. You need to place the sample data you use in `algorithmTemplate/algorithm_app/experiment/YOUR_USERNAME/origin_data`, where YOUR_USERNAME is any name you choose.

### Docker compose
- First, you need to install Docker Compose.
- Once you complete installation, execute the following command under `algorithmTemplate/`:
    ```bash
    docker compose build
    docker compose up -d
    ```
- After the container is successfully running, execute the command `curl http://localhost:18010/test?user=YOUR_USERNAME` to call the `test` interface, and you can observe the algorithm's operation.
