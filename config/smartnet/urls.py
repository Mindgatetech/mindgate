from django.urls import path
from . import views
urlpatterns = [
    path('', views.smartnet_http, name='smartnet'),
    path('hook', views.smartnet_hook, name='smartnet_hook'),
    path('import', views.import_test),
]