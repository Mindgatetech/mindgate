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
    path('aimodel/add', views.aimodel_add, name='aimodel_add'),
    path('aimodel/delete/<int:id>', views.aimodel_delete, name='aimodel_delete'),
]