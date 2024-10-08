from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages, auth
from django.db.models import Q
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from . import models
from mindnet.models import Paper, Dataset, AiModel, Preprocess, Metric, Scaler, ValidationTechnique, PipeJob
import json

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
        if 'profile' in request.POST:
            fullname    = request.POST.get('fullname')
            phone       = request.POST.get('phone')
            email       = request.POST.get('email')
            bio         = request.POST.get('bio')
            researcher  = bool(request.POST.get('researcher', False))
            user.fullname   = fullname
            user.phone      = phone
            user.email      = email
            user.bio        = bio
            user.researcher = researcher
            user.save()
            messages.success(request, 'Settings has updated successfully.')
            return redirect('settings')
        else:
            if len(request.FILES) != 0:
                    user.avatar = request.FILES['avatarfile']
                    user.save()
                    return redirect('settings')
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
        dataset.related_paper.clear()
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

@login_required
def aimodel_details(request, id):
    user = request.user
    if request.method == 'POST':
        if AiModel.objects.filter(user=user, pk=id).exists():     
            aimodel = AiModel.objects.get(user=user, pk=id)
            aimodel.name = request.POST.get('name')
            aimodel.description = request.POST.get('description')
            aimodel.framework = request.POST.get('framework')
            related_paper_id = request.POST.get('related_paper_id')
            aimodel.approach = request.POST.get('approach')
            aimodel.private = bool(request.POST.get('private', False))
            if len(request.FILES) != 0:
                aimodel.model = request.FILES['model']

            aimodel.related_paper.clear()
            if Paper.objects.filter(pk=related_paper_id).exists():
                paper = Paper.objects.filter(pk=related_paper_id)
                aimodel.related_paper.set(paper)
            aimodel.save()
            # message
            return redirect('/cpanel/aimodels/mine')
        # message you have not premission

        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


    queryset = AiModel.objects.filter(user=user, pk=id) | AiModel.objects.filter(private=False, pk=id)
    aimodel  = get_object_or_404(queryset, pk=id)
    papers = Paper.objects.filter(private=False) | Paper.objects.filter(user=user)
    context  = { 'aimodel': aimodel,
                 'papers' : papers,
                   #dataset
                    }
    return render(request, 'cpanel/aimodel/details.html', context=context)


@login_required()
def aimodel_add(request):
    user = request.user
    if request.method == 'POST':
        
        name            = request.POST.get('name')
        description     = request.POST.get('description')
        framework       = request.POST.get('framework')
        approach        = request.POST.get('approach')
        related_paper_id= request.POST.get('related_paper_id')
        model           = request.FILES['model']
        private         = bool(request.POST.get('private', False))
        aimodel = AiModel.objects.create(
            user=user, name=name, description=description,
              framework=framework, approach=approach, model=model,
                private=private)
        if Paper.objects.filter(pk=related_paper_id).exists():
            paper = Paper.objects.filter(pk=related_paper_id)
            aimodel.related_paper.set(paper)
            aimodel.save()
        return redirect('/cpanel/aimodels/mine')
    papers = Paper.objects.filter(private=False) | Paper.objects.filter(user=user)
    approach = AiModel.approach.field.choices
    framework= AiModel.framework.field.choices
    context = {
        'papers': papers,
        'approach': approach,
        'framework': framework,
    }
    return render(request, 'cpanel/aimodel/add.html', context=context)


@login_required()
def aimodel_delete(request, id):
    user = request.user
    AiModel.objects.filter(user=user, pk=id).delete()
    return redirect('/cpanel/aimodels/mine')

# Metric 

@login_required
def metrics(request, opt):
    if opt == 'public':
        metrics = Metric.objects.filter(private=False)
    else:
        user = request.user
        metrics = Metric.objects.filter(user=user)
    context  = {
        'metrics': metrics
    }
    return render(request, 'cpanel/metric/metrics.html', context=context)

@login_required
def metric_details(request, id):
    user = request.user
    if request.method == 'POST':
        if Metric.objects.filter(user=user, pk=id).exists():     
            metric = Metric.objects.get(user=user, pk=id)
            metric.name = request.POST.get('name')
            metric.description = request.POST.get('description')
            related_paper_id = request.POST.get('related_paper_id')
            metric.private = bool(request.POST.get('private', False))
            if len(request.FILES) != 0:
                metric.metric = request.FILES['metricfile']
            metric.related_paper.clear()
            if Paper.objects.filter(pk=related_paper_id).exists():
                paper = Paper.objects.filter(pk=related_paper_id)
                metric.related_paper.set(paper)
            metric.save()
            # message
            return redirect('/cpanel/metrics/mine')
        # message you have not premission

        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


    queryset = Metric.objects.filter(user=user, pk=id) | Metric.objects.filter(private=False, pk=id)
    metric  = get_object_or_404(queryset, pk=id)
    papers = Paper.objects.filter(private=False) | Paper.objects.filter(user=user)
    context  = { 'metric': metric,
                 'papers' : papers,
                   #dataset
                    }
    return render(request, 'cpanel/metric/details.html', context=context)

@login_required()
def metric_add(request):
    user = request.user
    if request.method == 'POST':
    
        name            = request.POST.get('name')
        description     = request.POST.get('description')
        related_paper_id= request.POST.get('related_paper_id')
        metricfile      = request.FILES['metricfile']
        private         = bool(request.POST.get('private', False))
        metric = Metric.objects.create(
            user=user, name=name, description=description,
              metric=metricfile, private=private)
        if Paper.objects.filter(pk=related_paper_id).exists():
            paper = Paper.objects.filter(pk=related_paper_id)
            metric.related_paper.set(paper)
            metric.save()
        return redirect('/cpanel/metrics/mine')
    papers = Paper.objects.filter(private=False) | Paper.objects.filter(user=user)
    context = {
        'papers': papers
    }
    return render(request, 'cpanel/metric/add.html', context=context)

@login_required()
def metric_delete(request, id):
    user = request.user
    Metric.objects.filter(user=user, pk=id).delete()
    return redirect('/cpanel/metrics/mine')

# Scaler 

@login_required
def scalers(request, opt):
    if opt == 'public':
        scalers = Scaler.objects.filter(private=False)
    else:
        user = request.user
        scalers = Scaler.objects.filter(user=user)
    context  = {
        'scalers': scalers
    }
    return render(request, 'cpanel/scaler/scalers.html', context=context)

@login_required
def scaler_details(request, id):
    user = request.user
    if request.method == 'POST':
        if Scaler.objects.filter(user=user, pk=id).exists():     
            scaler = Scaler.objects.get(user=user, pk=id)
            scaler.name = request.POST.get('name')
            scaler.description = request.POST.get('description')
            related_paper_id = request.POST.get('related_paper_id')
            scaler.private = bool(request.POST.get('private', False))
            if len(request.FILES) != 0:
                scaler.scaler = request.FILES['scalerfile']
            scaler.related_paper.clear()
            if Paper.objects.filter(pk=related_paper_id).exists():
                paper = Paper.objects.filter(pk=related_paper_id)
                scaler.related_paper.set(paper)
            scaler.save()
            # message
            return redirect('/cpanel/scalers/mine')
        # message you have not premission

        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


    queryset = Scaler.objects.filter(user=user, pk=id) | Scaler.objects.filter(private=False, pk=id)
    scaler  = get_object_or_404(queryset, pk=id)
    papers = Paper.objects.filter(private=False) | Paper.objects.filter(user=user)
    context  = { 'scaler': scaler,
                 'papers' : papers,
                   #dataset
                    }
    return render(request, 'cpanel/scaler/details.html', context=context)

@login_required()
def scaler_add(request):
    user = request.user
    if request.method == 'POST':
    
        name            = request.POST.get('name')
        description     = request.POST.get('description')
        related_paper_id= request.POST.get('related_paper_id')
        scalerfile      = request.FILES['scalerfile']
        private         = bool(request.POST.get('private', False))
        scaler = Scaler.objects.create(
            user=user, name=name, description=description,
              scaler=scalerfile, private=private)
        if Paper.objects.filter(pk=related_paper_id).exists():
            paper = Paper.objects.filter(pk=related_paper_id)
            scaler.related_paper.set(paper)
            scaler.save()
        return redirect('/cpanel/scalers/mine')
    papers = Paper.objects.filter(private=False) | Paper.objects.filter(user=user)
    context = {
        'papers': papers
    }
    return render(request, 'cpanel/scaler/add.html', context=context)

@login_required()
def scaler_delete(request, id):
    user = request.user
    Scaler.objects.filter(user=user, pk=id).delete()
    return redirect('/cpanel/scalers/mine')

# Pipe Job

@login_required
def pipejobs(request):
    user = request.user
    pipejobs = PipeJob.objects.filter(user=user)
    context  = {
        'pipejobs': pipejobs
    }
    return render(request, 'cpanel/pipejob/pipejobs.html', context=context)
    
@login_required
def pipejob_details(request, id):
    user        = request.user
    pipejob     = get_object_or_404(PipeJob, user=user, pk=id)
    context     = {
        'pipejob': pipejob 
                    }
    return render(request, 'cpanel/pipejob/details.html', context=context)

@login_required()
def pipejob_add(request):
    def validator(data):
        if data == '':
            return -1.0
        return float(data)
    user = request.user
    if request.method == 'POST':

        dataset_id      = request.POST.get('dataset_id')
        aimodel_id      = request.POST.get('aimodel_id')
        preprocess_id   = request.POST.get('preprocess_id')
        scaler_id       = request.POST.get('scaler_id')
        metric_id       = request.POST.get('metric_id')
        vt_id           = request.POST.get('vt_id')
        tmin            = validator(request.POST.get('tmin'))
        tmax            = validator(request.POST.get('tmax'))
        event_id        = request.POST.get('event_id')
        montage_type_id = request.POST.get('montage_type_id')
        filter_id       = request.POST.get('filter_id')
        low_band        = request.POST.get('low_band', 0)
        high_band       = request.POST.get('high_band', 120)
        epoch_from_id   = request.POST.get('epoch_from_id')
        duration        = validator(request.POST.get('duration'))
        overlap         = validator(request.POST.get('overlap'))
        event_from_id   = request.POST.get('event_from_id')
        on_missing_id   = request.POST.get('on_missing_id')
        stim_channel    = request.POST.get('stim_channel')
        eeg_channels    = request.POST.get('eeg_channels')
        eog_channels    = request.POST.get('eog_channels')
        exclude         = request.POST.get('exclude')
        baseline        = request.POST.get('baseline')
        projection      = bool(request.POST.get('projection', False))
        private         = bool(request.POST.get('private', False))
        dataset     = get_object_or_404(Dataset, pk=dataset_id)
        aimodel     = get_object_or_404(AiModel, pk=aimodel_id)
        preprocess  = get_object_or_404(Preprocess, pk=preprocess_id)
        scaler      = get_object_or_404(Scaler, pk=scaler_id)
        metric      = get_object_or_404(Metric, pk=metric_id)
        vt          = get_object_or_404(ValidationTechnique, pk=vt_id)
        
        if PipeJob.objects.create(
            user=user, dataset=dataset, aimodel=aimodel, preprocess=preprocess, scaler=scaler,
              metric=metric, validationtechnique=vt, tmin=tmin, tmax=tmax, epoch_from=epoch_from_id,
                duration=duration, overlap=overlap, event_from=event_from_id, event_id=event_id, filter=filter_id,
                  high_band=high_band, low_band=low_band, on_missing=on_missing_id, montage_type=montage_type_id, stim_channel=stim_channel,
                  eog_channels=eog_channels, eeg_channels=eeg_channels, exclude=exclude, baseline=baseline, projection=projection, private=private
                  ) :
            messages.success(request, 'Pipe Job has sent successfully.')
            pass
        else :
            messages.error(request, 'There is an error, try again...')
            pass    
        
        return redirect('pipejobs')
    datasets         = Dataset.objects.filter(private=False) | Dataset.objects.filter(user=user)
    aimodels         = AiModel.objects.filter(private=False) | AiModel.objects.filter(user=user)
    preprocess       = Preprocess.objects.filter(private=False) | Preprocess.objects.filter(user=user)
    scalers          = Scaler.objects.filter(private=False) | Scaler.objects.filter(user=user) 
    metrics          = Metric.objects.filter(private=False) | Metric.objects.filter(user=user)
    vts              = ValidationTechnique.objects.filter(private=False) | ValidationTechnique.objects.filter(user=user)
    montage_type     = PipeJob.montage_type.field.choices
    filter           = PipeJob.filter.field.choices
    epoch_from       = PipeJob.epoch_from.field.choices
    event_from       = PipeJob.event_from.field.choices
    on_missing       = PipeJob.on_missing.field.choices
    context = {
        'datasets':datasets, 'aimodels':aimodels, 'preprocess':preprocess, 
            'scalers':scalers, 'metrics':metrics, 'vts':vts, 'montage_type':montage_type,
              'filter':filter, 'epoch_from':epoch_from, 'event_from':event_from, 'on_missing':on_missing  
    }
    return render(request, 'cpanel/pipejob/add.html', context=context)

@login_required()
def pipejob_compare(request):
    user = request.user
    if request.method == 'POST':
        jobs_selection = request.POST.getlist('jobs_selection[]')
        if len(jobs_selection) > 0:
            query = Q(pk=jobs_selection[0])
            for pj in jobs_selection:
                query.add(Q(pk=pj), Q.OR)
            query.add(Q(status=True), Q.AND)
            pipejobs = PipeJob.objects.filter(query)
            if pipejobs.count() != len(jobs_selection):
                messages.error(request, 'Jobs seems to be incompleted !')
                return redirect('pipejobs')
            metricList = list()
            vtList     = list()
            for job in pipejobs:
                metricList.append(job.metric.name)
                axilaryList= list(set(metricList))
                metricList.pop()
                metricList = axilaryList.copy()
                vtList.append(job.validationtechnique.name)
                axilaryList.clear()
                axilaryList= list(set(vtList))
                vtList.pop()
                vtList = axilaryList.copy()
            if len(vtList)>1 or len(metricList)>1:
                messages.error(request, 'Metrics or Validation Tecniques are not same !')
                return redirect('pipejobs')
            context = {
                'pipejobs': pipejobs,
            }
            data_dict = dict()
            for pj in pipejobs:
                data = list()
                results = pj.results
                results = results.replace('array(', "'")
                results = results.replace(')', "'")
                results = results.replace("'", "\"")
                results = json.loads(results)
                last_key = next(reversed(results.keys()))
                results = results[last_key].replace("[", '')
                results = results.replace("]", '')
                results = results.split(',')
                '''for i,item in enumerate(results):
                    results[i] = float(item)'''
                avg = 0 
                for i, val in enumerate(results):
                    data.append({"label": str(i+1), "y": float(val)})
                    avg = avg + float(val)
                print(i, avg)
                data.append({"label": "AVG", "y": avg/(i+1) })
                data_dict[str(pj.id)] = data
                metric_name = last_key
                
            context = {
                'data_dict' : data_dict,
                'metric_name' : metric_name
            }
            return render(request, 'cpanel/pipejob/compare.html', context=context)
        
    messages.error(request, 'You have to select at least one choices')
    return redirect('pipejobs')

@login_required()
def pipejob_delete(request, id):
    user = request.user
    Scaler.objects.filter(user=user, pk=id).delete()
    return redirect('/cpanel/scalers/mine')
