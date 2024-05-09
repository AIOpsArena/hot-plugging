from django.views import View
from django.views.decorators.csrf import csrf_exempt
from algorithm_app.algorithm import Algorithm
from django.http import JsonResponse
import time
# Create your views here.

@csrf_exempt
class AlgorithmView(View):
    def model_run(request):
        if request.method == 'GET':
            user = request.GET.get('user')
        
            algorithm = Algorithm(user)
            
            # 运行算法
            algorithm.run()
            pass

            return JsonResponse({'data': {}, 'message': 'SUCCESS'}, status=200)
        
    def model_train(request):
        if request.method == 'GET':
            user = request.GET.get('user')
        
            algorithm = Algorithm(user)
            
            # 运行算法
            algorithm.train()
            pass

            return JsonResponse({'data': {}, 'message': 'SUCCESS'}, status=200)
        
    def model_test(request):
        if request.method == 'GET':
            user = request.GET.get('user')
        
            algorithm = Algorithm(user)
            
            # 运行算法
            algorithm.test()
            pass

            return JsonResponse({'data': {}, 'message': 'SUCCESS'}, status=200)
        
    def test(request):
        if request.method == 'GET':
            user = request.GET.get('user')
        
            algorithm = Algorithm(user)
            
            # test your algorithm here
            pass

            return JsonResponse({'data': {}, 'message': 'SUCCESS'}, status=200)