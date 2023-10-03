"""demo_django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from web import views

urlpatterns = [
    # path('admin/', admin.site.urls),
    path('test/', views.test),
    path('page1/', views.get_page1,name='page1'),
    path('page2/', views.get_page2,name='page2'),
    path('page3/', views.get_page3,name='page3'),
    path('page4/', views.get_page4,name='page4'),
    path('page1/button1', views.authorized),
    path('page2/button2', views.revision_space),
    path('page2/button3', views.generate),
    path('page3/button3', views.getResult),
    path('page4/button2', views.revision_device),
    path('page4/button3', views.getTransResult),
]
