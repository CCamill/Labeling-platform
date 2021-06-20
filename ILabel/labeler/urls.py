from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('get_predict_contours/', views.get_predict_contours, name='get_predict_contours'),
    path('download_zipfile/', views.download_zipfile, name='download_zipfile'),
    path('progressurl/', views.show_progress, name='progress'),
    path('save_current_image/', views.save_current_image, name='save_current_image'),
    path('download_patient_zipfile/',views.download_patient_zipfile,name='download_patient_zipfile'),
]