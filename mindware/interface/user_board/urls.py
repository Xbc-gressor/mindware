# License: MIT

from django.urls import path

from . import views

app_name = 'user_board'
urlpatterns = [
    # /user_board/
    path('', views.index, name='index'),
    path('index/', views.index, name='index'),

]
