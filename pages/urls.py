from django.urls import path
from .views import homePageView, aboutPageView, results, homePost


urlpatterns = [
    path('', homePageView, name='home'),
    path('about/', aboutPageView, name='about'),
    path('homePost/', homePost, name='homePost'),
    path('results/<int:blueDragons>/<int:blueKills>/<int:redDragons>/<int:redKills>', results, name='results'),

]
