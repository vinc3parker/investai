from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def performance(request):
    return render(request, 'performance.html')

def profile(request):
    return render(request, 'profile.html')