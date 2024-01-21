from django.db import models

class Result(models.Model):
    result_id = models.CharField(max_length=32)
    result = models.TextField(blank=True)

    def __str__(self):
        return self.result_id