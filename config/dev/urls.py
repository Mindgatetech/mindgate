from django.urls import path
from . import views
urlpatterns = [
    path('tokens', views.tokens, name='tokens'),
    path('token/delete/<int:id>', views.token_delete, name='token_delete'),
]