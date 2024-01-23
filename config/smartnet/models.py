import datetime, os
from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver

class Result(models.Model):
    related_result  = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)
    result_id       = models.CharField(max_length=32)
    time            = models.DateTimeField(default=datetime.datetime.now)
    results         = models.TextField(blank=True)
    best_models_path= models.TextField(blank=True)
    processed       = models.BooleanField(default=False)

    def __str__(self):
        return self.result_id

@receiver(post_delete, sender=Result)
def result_delete(sender, instance, **kwargs):
    best_models_path = instance.best_models_path
    bmp = [path for path in best_models_path.split(',')]
    for path in bmp:
        os.remove('media/'+path)


