from django.contrib import admin
from . import models
# Register your models here.


class ResultAdmin(admin.ModelAdmin):
    list_display = ['result_id', 'time', 'processed']

admin.site.register(models.Result, ResultAdmin)