from django.urls import path
from . import views
urlpatterns = [
    path('', views.Scheduled_test),
    path('import', views.import_test),
    path('hook', views.hook_test),
]