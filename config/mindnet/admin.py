from django.contrib import admin, messages
from . import models
# Register your models here.



class PaperAdmin(admin.ModelAdmin):
    list_display = ('title','private')

class DatasetAdmin(admin.ModelAdmin):
    list_display = ('name','type','private','ready_to_use')
    actions = ("uppercase",)  # Necessary
    @admin.action(description='Make selected name uppercase')
    def uppercase(modeladmin, request, queryset):
        for obj in queryset:
            obj.name = obj.name.upper()
            obj.save()
            messages.success(request, "Successfully made uppercase!")

class PipeJobAdmin(admin.ModelAdmin):
    list_display = ('aimodel','evaluation_type','metric','status')

admin.site.register(models.Paper, PaperAdmin)
admin.site.register(models.Dataset, DatasetAdmin)
admin.site.register(models.AiModel)
admin.site.register(models.Preprocess)
admin.site.register(models.ValidationTechnique)
admin.site.register(models.Scaler)
admin.site.register(models.Metric)
admin.site.register(models.PipeJob, PipeJobAdmin)

