# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from .models import Records
from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import json
from django.core.serializers.json import DjangoJSONEncoder
# Create your views here.



def getData(request):
   
    
    # labels = []
    # data = []

    queryset = Records.objects.all().values('gender','ageGroup')
    data=json.dumps(list(queryset), cls=DjangoJSONEncoder)
    # print(data)
    # for value in queryset:
    #     labels.append(value.gender)
    #     data.append(value.ageGroup)
    #     print(queryset)
    # print(labels)
    # print(data)
    return render(request,'chart-details.html', {"data":data} )
    # return render(request,'face-details.html')
    