import requests
from django.shortcuts import render
from django.http import JsonResponse

# Create your views here.
def home(request):
    return render(request, 'home.html')

def get_tickers(request):
    url = "http://127.0.0.1:5000/tickers"
    response = requests.get(url)
    tickers = response.json()
    return JsonResponse(tickers, safe=False)

def fetch_stock_data(request, ticker):
    url = f"http://127.0.0.1:5000/fetch/{ticker}"
    response = requests.get(url)
    data = response.json()
    return JsonResponse(data, safe=False)

def stocks(request):
    return render(request, 'stocks.html')

def performance(request):
    return render(request, 'performance.html')

def profile(request):
    return render(request, 'profile.html')