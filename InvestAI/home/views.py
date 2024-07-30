import requests
from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'home.html')

def about(request):
    # URL of the Database Flask API
    ticker = 'aapl'
    url = f"http://127.0.0.1:5000/fetch/{ticker}"

    # Make a GET request to the Flast API
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json() # Convert the response data to JSON
    else: 
        data = None
    
    # Render teh response data in a Django template
    return render(request, 'about.html', {'data': data})

def performance(request):
    return render(request, 'performance.html')

def profile(request):
    return render(request, 'profile.html')