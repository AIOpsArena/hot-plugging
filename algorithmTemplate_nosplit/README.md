# Test your algorithm

1. First, build docker image. Run the following command:

    cd algorithmTemplate_split/

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


