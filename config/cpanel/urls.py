from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('settings', views.settings, name='settings'),
    # Auth
    path('register', views.dashboard_register, name='register'),
    path('login', views.dashboard_login, name='login'),
    path('logout', views.dashboard_logout, name='logout'),
    # Paper
    path('papers/<str:opt>', views.papers, name='papers'),
    path('paper/add', views.paper_add, name='paper_add'),
    path('paper/delete/<int:id>', views.paper_delete, name='paper_delete'),

    # Dataset
    path('datasets/<str:opt>', views.datasets, name='datasets'),    
    path('dataset/<int:id>', views.dataset_details, name='dataset_details'),
    path('dataset/add', views.dataset_add, name='dataset_add'),
    path('dataset/delete/<int:id>', views.dataset_delete, name='dataset_delete'),
    
    # AiModel
    path('aimodels/<str:opt>', views.aimodels, name='aimodels'),    
    path('aimodel/<int:id>', views.aimodel_details, name='aimodel_details'),
    path('aimodel/add', views.aimodel_add, name='aimodel_add'),
    path('aimodel/delete/<int:id>', views.aimodel_delete, name='aimodel_delete'),

    # Metric
    path('metrics/<str:opt>', views.metrics, name='metrics'),    
    path('metric/<int:id>', views.metric_details, name='metric_details'),
    path('metric/add', views.metric_add, name='metric_add'),
    path('metric/delete/<int:id>', views.metric_delete, name='metric_delete'),

    # Scaler
    path('scalers/<str:opt>', views.scalers, name='scalers'),    
    path('scaler/<int:id>', views.scaler_details, name='scaler_details'),
    path('scaler/add', views.scaler_add, name='scaler_add'),
    path('scaler/delete/<int:id>', views.scaler_delete, name='scaler_delete'),

    # PipeJob
    path('pipejobs', views.pipejobs, name='pipejobs'),    
    path('pipejob/<int:id>', views.pipejob_details, name='pipejob_details'),
    path('pipejob/add', views.pipejob_add, name='pipejob_add'),
    path('pipejob/delete/<int:id>', views.pipejob_delete, name='pipejob_delete'),
]