import shutil, datetime, os
from . import views
from django.db import models
from django.dispatch import receiver
from django_q.tasks import async_task
from cpanel.models import User
from django.db.models.signals import post_save, post_delete

# Paper Model
class Paper(models.Model):
    user  = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user_paper')
    title = models.CharField(max_length=200, default=None)
    abstract = models.TextField(default=None)
    published_year = models.IntegerField(default=None)
    doi = models.CharField(max_length=300, default=None)
    private = models.BooleanField(default=True)

    def __str__(self):
        return self.title

# Dataset Model
class Dataset(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user_dataset')
    name            = models.CharField(max_length=25, default=None)
    description     = models.TextField(default=None, blank=True)
    TYPE_CHOICE     = [('EEG','EEG'), ('fMRI','fMRI')]
    type            = models.CharField(choices=TYPE_CHOICE, default='EEG', max_length=10)
    FIELD_CHOICE    = [('MI','Motor Imagery'), ('ERP','ERP')]
    research_field  = models.CharField(choices=FIELD_CHOICE, default='MI', max_length=20)
    channels        = models.CharField(max_length=256, blank=True, null=True)
    #eog_channels    = models.CharField(max_length=256, blank=True, null=True)
    dataset_link    = models.URLField(blank=False, null=True, max_length=500)
    dataset_path    = models.CharField(max_length=256, blank=True, null=True)
    related_paper   = models.ManyToManyField('Paper', blank=True, default=None, null=True)
    #related_models  = models.ManyToManyField('AiModel', blank=True, default=None)
    extracted_file_path = models.TextField(default=None, null=True, blank=True)
    metadata        = models.TextField(null=True, blank=True)
    private         = models.BooleanField(default=False)
    ready_to_use    = models.BooleanField(default=False)

    def __str__(self):
        return self.name
@receiver(post_save, sender=Dataset)
def dataset_processing(sender, instance, created, **kwargs):
    if created:
        model         = instance
        opts = {
        'group':'Dataset Processing',
        }
        async_task(views.dataset_processor, model, q_options=opts)

@receiver(post_delete, sender=Dataset)
def dataset_delete(sender, instance,  **kwargs):
    efp = instance.extracted_file_path
    data_path = [fp for fp in efp.split(',')]
    for fp in data_path:
        os.remove(fp)

# Ai Model
class AiModel(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user_aimodel')
    name            = models.CharField(max_length=25,)
    description     = models.TextField(default=None, blank=True)
    Framework_CHOICE= [('Scikit-learn','Scikit-learn'),('Keras','Keras'), ('Pytorch','Pytorch')]
    framework       = models.CharField(choices=Framework_CHOICE, default='Scikit-learn', max_length=15)
    Approach_CHOICE = [('DL','Deep Learning'), ('ML','Machine Learning')]
    approach        = models.CharField(choices=Approach_CHOICE, default='ML', max_length=20)
    model           = models.FileField(upload_to='Ai_models/', default=None, blank=True)
    model_code      = models.TextField(null=True, blank=True)
    hyperparameters = models.TextField(null=True, blank=True)
    related_paper   = models.ManyToManyField('Paper',blank=True, null=True)
    related_dataset = models.ManyToManyField('Dataset', blank=True)
    granted         = models.BooleanField(default=False)
    private         = models.BooleanField(default=False)

    def __str__(self):
        return self.name

# Preprocess Model
class Preprocess(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user_preprocess')
    name            = models.CharField(max_length=25,)
    description     = models.TextField(default=None, blank=True)
    preprocess_code = models.TextField(null=True, blank=True)
    hyperparameters = models.TextField(null=True, blank=True)
    related_paper   = models.ManyToManyField('Paper',blank=True)
    private         = models.BooleanField(default=False)

    def __str__(self):
        return self.name

# Scaler Model
class Scaler(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user_scaler')
    name            = models.CharField(max_length=25,)
    description     = models.TextField(default=None, blank=True)
    scaler          = models.FileField(upload_to='Scalers/', null=True, blank=True)
    hyperparameters = models.TextField(null=True, blank=True)
    related_paper   = models.ManyToManyField('Paper',blank=True)
    granted         = models.BooleanField(default=False)
    private         = models.BooleanField(default=False)

    def __str__(self):
        return self.name

# Cross-validation Model
class CrossValidation(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user_cross_validation')
    name            = models.CharField(max_length=25,)
    description     = models.TextField(default=None, blank=True)
    cv_code         = models.TextField(null=True, blank=True)
    hyperparameters = models.TextField(null=True, blank=True)
    related_paper   = models.ManyToManyField('Paper',blank=True)
    private         = models.BooleanField(default=False)

    def __str__(self):
        return self.name
# Metric Model
class Metric(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user_metric')
    name            = models.CharField(max_length=25,)
    description     = models.TextField(default=None, blank=True)
    metric          = models.FileField(upload_to='Metrics/', null=True, blank=True)
    #hyperparameters = models.TextField(null=True, blank=True)
    related_paper   = models.ManyToManyField('Paper',blank=True)
    granted         = models.BooleanField(default=False)
    private         = models.BooleanField(default=False)

    def __str__(self):
        return self.name

# Pipe-Job Model
class PipeJob(models.Model):
    user            = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user_pipejob')
    EVALUATION_OPT  = [('SI', 'Subject Independent'), ('SD', 'Subject Dependent')]
    evaluation_type = models.CharField(choices=EVALUATION_OPT, default='standard_1020', max_length=2)
    dataset         = models.ForeignKey('Dataset', on_delete=models.CASCADE, null=True)
    preprocess      = models.ForeignKey('Preprocess', on_delete=models.CASCADE, null=True)
    scaler          = models.ForeignKey('Scaler', on_delete=models.CASCADE, null=True)
    metric          = models.ForeignKey('Metric', on_delete=models.CASCADE, null=True)
    crossvalidation = models.ForeignKey('Crossvalidation', on_delete=models.CASCADE, null=True)
    aimodel         = models.ForeignKey('AiModel', on_delete=models.CASCADE, null=True)
    test_size       = models.FloatField(max_length=5, default=0.3)
    tmin            = models.FloatField(max_length=5, default=0.5)
    tmax            = models.FloatField(max_length=5, default=2.5)
    event_id        = models.CharField(max_length=30, default="7,8")
    MONTAGE_OPT     = [('None', 'None'), ('standard_1005', 'standard_1005'), ('standard_1020', 'standard_1020')]
    montage_type    = models.CharField(choices=MONTAGE_OPT, default='None', max_length=20)
    FILTER_OPT      = [(None, 'None'), ('Bandpass', 'Bandpass')]
    filter          = models.CharField(choices=FILTER_OPT, default='Bandpass', max_length=20)
    low_band        = models.FloatField(default=8.0)
    high_band       = models.FloatField(default=35.0)
    EVENT_OPT       = [('A', 'Annotations'), ('SC', 'Stim Channel')]
    event_from      = models.CharField(max_length=2, choices=EVENT_OPT, default='Annotations')
    MISSING_OPT     = [('warn', 'warn'), ('ignore', 'Ignore'), ('raise', 'Raise')]
    on_missing      = models.CharField(max_length=12, choices=MISSING_OPT, default='warn')
    stim_channel       = models.CharField(max_length=100, blank=True)
    eeg_channels    = models.CharField(max_length=200, blank=True)
    eog_channels    = models.CharField(max_length=200, default="'EOG:ch01','EOG:ch02','EOG:ch03'")
    exclude         = models.CharField(max_length=20, default='bads')
    projection      = models.BooleanField(default=True)
    baseline        = models.CharField(max_length=20, default='None')
    start_time      = models.DateTimeField(default=datetime.datetime.now, null=True)
    end_time        = models.DateTimeField(null=True, blank=True)
    status          = models.BooleanField(default=False)
    output_shape    = models.CharField(max_length=50, default="(1000,3,250)")
    results         = models.TextField(blank=True)
    trained_model   = models.FileField(blank=True, upload_to="models_trained")
    private         = models.BooleanField(default=False)
    job_id          = models.CharField(max_length=33, blank=True)

    def __str__(self):
        return self.user.username


@receiver(post_save, sender=PipeJob)
def job_processing(sender, instance, created, **kwargs):
    if created:
        model = instance
        opts = {
            'group': 'PipeJob Processing',
        }
        async_task(views.pipjob_processing, model, q_options=opts)