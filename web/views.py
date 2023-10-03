import os.path
import pickle
import re

from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
from RDBLab.getResult import get_result
from RDBLab.getTransferResult import get_transfer_result
from RDBLab.main import main


def test(request):
    result = get_result()
    print(result)

    context = {
        'result': result
    }
    return render(request, "test.html", context)


def get_page1(request):
    return render(request, 'page1.html')


def get_page2(request):
    with open("./config/conf.yaml", 'r+') as f:
        conf = f.readlines()
    context = {
        'conf': conf
    }
    return render(request, 'page2.html', context)


def get_page3(request):
    return render(request, 'page3.html')

def get_page4(request):
    return render(request, 'page4.html')


# page1/button1
def authorized(request):
    with open("./config/conf.yaml", 'w+') as f:
        f.write(request.GET['yml'])

    return HttpResponse()


# page2/button2
def revision_space(request):
    with open("./config/workload.yaml", 'r+') as f:
        conf = f.read()
    return HttpResponse(conf)


# page2/button3
def generate(request):
    context = request.GET['yml']
    # context = ' '.join(request.GET['yml'].split())
    # context = context.replace("; ", ";\n")
    with open("./config/conf.yaml", 'w+') as f:
        f.write(context)

    main()
    return HttpResponse()


# page3/button3
def getResult(request):
    with open("./config/knob.yaml", 'w+') as f:
        f.write(request.GET['yml'])

    pattern = re.compile(r'conf(\d*):')
    list = pattern.findall(request.GET['yml'])
    a = list[0]
    b = list[1]
    result = get_result(a, b)

    return HttpResponse(result)

# page4/button2
def revision_device(request):
    with open("./config/device.yaml", 'r+') as f:
        conf = f.read()
    return HttpResponse(conf)

# page4/button3
def getTransResult(request):
    with open("./config/device.yaml", 'w+') as f:
        f.write(request.GET['yml'])

    pattern = re.compile(r'conf: (\d*)')
    list = pattern.findall(request.GET['yml'])
    print(list)
    result = get_transfer_result(list[0])

    return HttpResponse(result)