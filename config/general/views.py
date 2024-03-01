from django.shortcuts import render

def index(request):

    context = {

    }
    return render(request, 'general/index.html', context=context)
