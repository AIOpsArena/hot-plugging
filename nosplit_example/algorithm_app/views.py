import time
from multiprocessing import Process
import multiprocessing

from django.http import JsonResponse
from django.utils import timezone
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from .algorithm import Algorithm
from .models import AlgorithmAppTaskexecute as TaskExecute
import json
import pandas as pd
import mysql.connector
import pathlib
from yaml import full_load
import sys
import os
root_path = pathlib.Path(__file__).parent.parent
sys.path.append(root_path)
# 读取配置文件
root_config = full_load(open(root_path / "config.yaml", "r"))
# Create your views here.

def create_pool(pool_name):
    pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name=pool_name,
        pool_size=10,  # 池中保持的连接数
        pool_reset_session=True,
        host=root_config['host'],
        database=root_config['database'],
        user=root_config['user'],
        password=root_config['password']
    )
    return pool

algorithm_pool = create_pool('algorithm')

def get_connection(pool):
    return pool.get_connection()

def execute(user, task_id):
    algorithm = Algorithm(user)

    try:
        result_dict = algorithm.run()
        execute_status = 'finished'
    except Exception as e:
        print(e)
        execute_status = 'failed'
    
    connection = get_connection(algorithm_pool)

    execute_result = json.dumps(result_dict)
    end_time = timezone.now()
    cursor = connection.cursor()
    cursor.execute("UPDATE algorithm_app_taskexecute SET execute_status = %s, pid = %s, container_pid = %s, end_time = %s, execute_result = %s WHERE id = %s", (execute_status, None, None, end_time, execute_result, task_id))
    connection.commit()

    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM algorithm_app_taskexecute WHERE id = %s", (task_id,))
    task = cursor.fetchone()
    template_id = task['template_id']
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM algorithm_app_tasktemplate WHERE id = %s", (template_id,))
    template = cursor.fetchone()

    if template['is_evaluate'] == 'True':
        # 更新record
        algorithm_id = template['algorithm_id']
        record_id = template['record_id']

        cursor = connection.cursor()

        # Fetch current result from the database
        cursor.execute("SELECT result FROM leaderboard_app_leaderboardrecord WHERE id = %s", (record_id,))
        current_result_json = cursor.fetchone()[0]
        if current_result_json:
            current_result = json.loads(current_result_json)
        else:
            current_result = {}
        # Update result
        current_result[str(algorithm_id)] = result_dict

        # Update the record with new result
        cursor.execute("UPDATE leaderboard_app_leaderboardrecord SET result = %s WHERE id = %s", 
                        (json.dumps(current_result), record_id))
        connection.commit()

    cursor.close()
    connection.close()

@csrf_exempt
class AlgorithmView(View):
    def model_run(request):
        if request.method == 'GET':
            user = request.GET.get('user')
            task_id = request.GET.get('id')
            process = Process(target=execute, args=(user, task_id))
            process.start()
            
            start_time = timezone.now()
            connection = get_connection(algorithm_pool)
            cursor = connection.cursor()
            cursor.execute("UPDATE algorithm_app_taskexecute SET container_pid = %s, start_time = %s WHERE id = %s", (process.pid, start_time, task_id))
            connection.commit()
            cursor.close()
            connection.close()
            return JsonResponse(data={}, status=200)
    def model_stop(request):
        if request.method == 'GET':
            try:
                user = request.GET.get('user')
                pid = int(request.GET.get('pid'))
                active_children = multiprocessing.active_children()
                p = None

                # 遍历列表查找与指定PID匹配的子进程对象
                for child in active_children:
                    if child.pid == pid:
                        p = child
                if p:
                    p.terminate()
            except Exception as e:
                print(f"中断容器失败!{e}")
            return JsonResponse({'data': {}, 'message': f'已杀死pid为{pid}的子进程'}, status=200)
    def test(request):
        if request.method == 'GET':
            user = request.GET.get('user')
        
            algorithm = Algorithm(user)
            
            # test your algorithm here
            pass

            return JsonResponse({'data': {}, 'message': 'SUCCESS'}, status=200)