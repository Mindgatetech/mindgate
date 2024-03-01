from django.urls import path
from . import views
urlpatterns = [
    path('tokens', views.tokens, name='tokens'),
    path('token/delete/<int:id>', views.token_delete, name='token_delete'),
    path('auth_tokens', views.auth_token, name='auth_tokens'),
]