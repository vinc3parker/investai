from django.urls import path
from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path('stocks/', views.stocks, name='about'),
    path('api/tickers/', views.get_tickers, name='get_tickers'),
    path('api/ticker-data/<str:ticker>/', views.fetch_stock_data, name='fetch_stock_data'),
    path('api/predict/<str:ticker>/', views.get_stock_prediction, name='get_stock_prediction'),
    path('performance/', views.performance, name='performance'),
    path('profile/', views.profile, name='profile'),
]