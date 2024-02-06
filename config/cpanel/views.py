from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages, auth
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from . import models
from mindnet.models import Dataset, Paper, AiModel

@login_required()
def dashboard(request):
    return render(request, 'cpanel/index.html')

def dashboard_login(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method =='POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            return redirect('dashboard')
        else:
            return redirect('login')
    return render(request, 'cpanel/auth/login.html')

def dashboard_register(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        fullname = request.POST['fullname']
        email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        repassword = request.POST['repassword']
        phone = request.POST['phone']

        if len(password) | len(repassword) < 8 :
            messages.error(request, 'تعداد کاراکتر رمز عبور شما باید بیشتر از 8 باشد')
            return redirect('register')
        else:
            if password != repassword : 
                messages.error(request, 'تکرار رمز عبور با رمز عبور برابر نیست')
                return redirect('register')
            else:
                if  models.User.objects.filter(username=username).exists():
                    messages.error(request, 'این نام کاربری قبلا گرفته شده است')
                    return redirect('register')
                else:
                    if models.User.objects.filter(email=email).exists():
                        messages.error(request, 'این ایمیل قبلا ثبت نام کرده است')
                        return redirect('register')
                    else:
                        user = models.User.objects.create_user(
                            username=username, password=password, email=email, fullname=fullname, phone=phone, researcher=True
                        )
                        user.save()
                        messages.success(request, 'تبریک ... ثبت نام با موفقیت انجام شد. وارد شوید')
                        return redirect('login')
                        
            return redirect('register')
    else:
        return render(request, 'cpanel/auth/register.html')

@login_required
def dashboard_logout(request):
    #if request.method == 'POST':
    auth.logout(request)
    messages.success(request, 'با موفقیت خارج شدید')
    return redirect('index')

@login_required()
def settings(request):
    user_id = request.user.id
    user = get_object_or_404(models.User, pk=user_id)
    if request.method == 'POST':
        if request.POST.get('profile'):
            pass
        else:
            pass
    
    context = {
        'user' : user,
    }
    return render(request, 'cpanel/settings.html', context=context)
    
@login_required
def papers(request, opt):
    user = request.user
    if opt == 'public':
        papers = Paper.objects.filter(private=False)
    else:
        papers = Paper.objects.filter(user=user)
    context = {
        'papers': papers,
    }
    return render(request, 'cpanel/paper/papers.html', context=context)

@login_required
def paper_add(request):
    if request.method == 'POST':
        user            = request.user
        title           = request.POST.get('title')
        abstract        = request.POST.get('abstract')
        published_year  = request.POST.get('published_year')
        doi             = request.POST.get('doi')
        private         = bool(request.POST.get('private', False))
        Paper.objects.create(user=user, title=title, abstract=abstract, published_year=published_year, doi=doi, private=private)
        return redirect('/cpanel/papers/mine')

    return render(request, 'cpanel/paper/add.html')

@login_required
def paper_delete(request, id):
    user = request.user
    Paper.objects.filter(user=user, pk=id).delete()
    return redirect('/cpanel/papers/mine')

# Dataset views

@login_required
def datasets(request, opt):
    if opt == 'public':
        datasets = Dataset.objects.filter(private=False)
    else:
        user = request.user
        datasets = Dataset.objects.filter(user=user)
    context  = {
        'datasets': datasets
    }
    return render(request, 'cpanel/dataset/datasets.html', context=context)

@login_required
def dataset_details(request, id):
    user = request.user
    if request.method == 'POST':
        if Dataset.objects.filter(user=user, pk=id).exists():     
            dataset = Dataset.objects.get(user=user, pk=id)
            dataset.name = request.POST.get('name')
            dataset.description = request.POST.get('description')
            dataset.type = request.POST.get('type')
            related_paper_id = request.POST.get('related_paper_id')
            if Paper.objects.filter(pk=related_paper_id).exists():
                paper = Paper.objects.filter(pk=related_paper_id)
                dataset.related_paper.set(paper)
            dataset.research_field = request.POST.get('research_field')
            dataset.channels = request.POST.get('channels')
            dataset.private = bool(request.POST.get('private', False))
            dataset.save()
            # message
            return redirect('/cpanel/datasets/mine')
        # message you have not premission

        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


    queryset = Dataset.objects.filter(user=user, pk=id) | Dataset.objects.filter(private=False, pk=id)
    dataset  = get_object_or_404(queryset, pk=id)
    papers = Paper.objects.filter(private=False) | Paper.objects.filter(user=user)
    context  = { 'dataset': dataset, 'papers' : papers }
    return render(request, 'cpanel/dataset/details.html', context=context)

@login_required
def dataset_add(request):
    user = request.user
    if request.method == 'POST':
        name            = request.POST.get('name')
        description     = request.POST.get('description')
        type            = request.POST.get('type')
        research_field  = request.POST.get('research_field')
        channels        = request.POST.get('channels')
        dataset_link    = request.POST.get('dataset_link')
        related_paper_id= request.POST.get('related_paper_id') 
        private         = bool(request.POST.get('private', False))
        dataset = Dataset.objects.create(user=user, name=name, description=description, type=type,
                                research_field=research_field, channels=channels,
                                  dataset_link=dataset_link, private=private)
        if Paper.objects.filter(pk=related_paper_id).exists():
                paper = Paper.objects.filter(pk=related_paper_id)
                dataset.related_paper.set(paper)
                dataset.save()
        #message 
        return redirect('/cpanel/datasets/mine')
    papers = Paper.objects.filter(private=False) | Paper.objects.filter(user=user)
    context = {
        'papers': papers
    }
    return render(request, 'cpanel/dataset/add.html', context=context)

@login_required
def dataset_delete(request, id):
    user = request.user
    Dataset.objects.filter(user=user, pk=id).delete()
    return redirect('/cpanel/datasets/mine')


# Ai Model

@login_required
def aimodels(request, opt):
    if opt == 'public':
        aimodels = AiModel.objects.filter(private=False)
    else:
        user = request.user
        aimodels = AiModel.objects.filter(user=user)
    context  = {
        'aimodels': aimodels
    }
    return render(request, 'cpanel/aimodel/aimodels.html', context=context)


@login_required()
def aimodel_add(request):
    if request.method == 'POST':
        user = request.user
        name            = request.POST.get('name')
        description     = request.POST.get('description')
        type            = request.POST.get('type')
        research_field  = request.POST.get('research_field')
        channels        = request.POST.get('channels')
        dataset_link    = request.POST.get('dataset_link')
        related_paper_id= request.POST.get('related_paper_id')
        related_paper   = Paper.objects.get(pk=related_paper_id)
        private         = request.POST.get('private')
        Dataset.objects.create(user, name, description, type, research_field, channels, dataset_link, related_paper, private)
        return redirect('aimodels', args=('mine',))
    papers = AiModel.objects.filter(private=False)
    context = {
        'papers': papers
    }
    return render(request, 'cpanel/aimodel/add.html', context=context)


@login_required()
def aimodel_delete(request, id):
    user = request.user
    AiModel.objects.filter(user=user, pk=id).delete()
    return redirect('aimodels', args=('mine',))