from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
class AlgorithmAppTaskexecute(models.Model):
    task_name = models.CharField(max_length=100, blank=True, null=True)
    dataset_range = models.CharField(max_length=100, blank=True, null=True)
    execute_type = models.IntegerField()
    execute_result = models.TextField(blank=True, null=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(blank=True, null=True)
    create_person = models.CharField(max_length=100, blank=True, null=True)
    create_time = models.DateTimeField()
    template = models.ForeignKey('AlgorithmAppTasktemplate', models.DO_NOTHING)
    execute_person = models.CharField(max_length=100, blank=True, null=True)
    execute_status = models.CharField(max_length=100, blank=True, null=True)
    pid = models.CharField(max_length=100, blank=True, null=True)
    container_pid = models.CharField(max_length=100, blank=True, null=True)
    train_or_test = models.CharField(max_length=100, blank=True, null=True)

    class Meta:
        managed = True
        db_table = 'algorithm_app_taskexecute'


class AlgorithmAppTasktemplate(models.Model):
    template_name = models.CharField(max_length=100)
    indicator_id = models.CharField(max_length=100)
    create_person = models.CharField(max_length=100)
    create_time = models.DateTimeField()
    algorithm = models.ForeignKey('AlgorithmAppAlgorithm', models.DO_NOTHING)

    class Meta:
        managed = True
        db_table = 'algorithm_app_tasktemplate'

class AlgorithmAppAlgorithm(models.Model):
    id = models.AutoField(primary_key=True)
    algorithm_name = models.CharField(max_length=100)
    algorithm_type = models.ForeignKey('AlgorithmAppAlgorithmType', on_delete=models.CASCADE, null=True, blank=False)
    indicator_id = models.CharField(max_length=100, null=True)
    cpu_count = models.CharField(max_length=100, null=True)
    mem_limit = models.CharField(max_length=100, null=True)
    container_created = models.BooleanField()
    container_status = models.BooleanField(null=True)
    container_id = models.CharField(max_length=100, null=True)
    container_port = models.CharField(max_length=100, null=True)
    container_ip = models.CharField(max_length=100, null=True)
    dataset_path = models.CharField(max_length=300, null=True)
    is_split = models.BooleanField()
    run_command = models.CharField(max_length=100, null=True)
    train_command = models.CharField(max_length=100, null=True)
    test_command = models.CharField(max_length=100, null=True)
    dataset_type = models.CharField(max_length=100, null=True)

    class Meta:
        managed = True
        db_table = 'algorithm_app_algorithm'

class AlgorithmAppAlgorithmType(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=100)

    class Meta:
        managed = True
        db_table = 'algorithm_app_algorithmtype'