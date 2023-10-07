from django.contrib import admin
from django.urls import path, include
from lawyer_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('lawyer/', include('lawyer_app.urls')),  # Include your app's URLs here
    path('predict_best_lawyer/', views.predict_best_lawyer, name='predict_best_lawyer'),
]
