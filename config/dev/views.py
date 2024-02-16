from django.shortcuts import render, get_list_or_404, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from . import models

@login_required
def tokens(request):
    user = request.user
    if request.method == 'POST':
        name        = request.POST.get('name')
        subject     = request.POST.get('subject')
        description = request.POST.get('description')
        models.Token.objects.create(user=user, name=name, description=description, subject=subject)
        messages.success(request, 'Your request has been sent for making decisions')
        return redirect('tokens')
    tokens  = models.Token.objects.filter(user=user)
    subjects= models.Token.subject.field.choices
    context = {
        'tokens' : tokens,
        'subjects' : subjects
    }
    return render(request, 'cpanel/dev/dev.html', context=context)

@login_required
def token_delete(request, id):
    user = request.user
    get_object_or_404(models.Token, user=user, pk=id).delete()
    messages.success(request, 'Your token has been deleted')
    return redirect('tokens')