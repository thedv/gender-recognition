"""faceRecog URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from faceRecog import views as app_views
from django.urls import include, path
from django.http import StreamingHttpResponse
from faceRecog.views import detect,gen,VideoCamera, video_feed
from django.http import HttpResponse
from rest_framework.response import Response
from django.shortcuts import render

urlpatterns = [
    # path('',app_views.index),
    # path('error_image',app_views.errorImg),
    # path('detect',app_views.detect),
    # path('detect_image',app_views.detectImage),
    # path('admin/', admin.site.urls),
    # path('records/', include('records.urls'))
    url(r'^$', app_views.index, name='index'),
    url(r'^error_image$', app_views.errorImg),
    url(r'^detect$', app_views.detect),
    url(r'^detect_image$', app_views.detectImage),
    url(r'^admin/', admin.site.urls),
    # path('', app_views.index1, name='index'),
    path('video_feed', app_views.video_feed, name='video_feed'),
    url(r'^records/', include('records.urls')),
    path('monitor/', lambda r: StreamingHttpResponse(gen(VideoCamera()),
                                                     content_type='multipart/x-mixed-replace; boundary=frame')),
]
