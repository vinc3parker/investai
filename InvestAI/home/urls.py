from django.urls import path
from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path('stocks/', views.stocks, name='about'),
    path('performance/', views.performance, name='performance'),
    path('profile/', views.profile, name='profile'),
]