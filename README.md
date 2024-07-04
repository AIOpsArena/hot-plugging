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

If you want to deploy the algorithm online to the platform, you will need to implement the __init__, train, and test methods. If there are intermediate results that need to be saved, the corresponding fields in the configuration file should be placed in the 'ignore_list' in 'algorithm_app/public_function.py'.

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
We have provided two example files, one with split train-test `split_example` and one without split train-test `nosplit_example`, hoping to be of some help to you.

Feel free to adapt this example to your specific needs and customize your algorithms for MicroServo platform. 🎉🎉

## Test your algorithm
After you have completed all operations, don't rush to upload; you can first test it locally. 😎

1. First, build docker image. Run the following command:

    docker build -t YOUR_IMAGE_NAME .

2. Second, run algorithm container.
   
   docker run --name 'YOUR_CONTAINER_NAME' -v $(pwd):/code YOUR_IMAGE_NAME python manage.py runserver 0.0.0.0:8000

3. Third, inspect container's ip

    docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' YOUR_CONTAINER_NAME

4. Fourth, implement the interface of algorithm_app/views.py:test. Such as algorithm.train() and algorithm.test().

5. Fifth, copy the demo dataset to algorithm_app/experiment.

    Create a USER folder under algorithm_app/experiment, and place the demo dataset in USER/origin_data

6. Sixth, send the curl request.

    curl "http:/CONTAINER_IP:8000/test?user=USER"