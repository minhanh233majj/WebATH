from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings


urlpatterns = [
    path('', views.home, name="home"),
    path('register/', views.register, name="register"),
    path('login/', views.loginPage, name="login"),
    path('search/', views.search, name="search"),
    path('category/', views.category, name="category"),
    path('about/', views.about, name="about"),
    path('detail/', views.detail, name="detail"),
    path('logout/', views.logoutPage, name="logout"),
    path('cart/', views.cart, name="cart"),
    path('checkout/', views.checkout, name="checkout"),
    path('update_item/', views.updateItem, name="update_item"),
    path('my-orders/', views.my_orders, name='my_orders'),
    path('lien-he/', views.lien_he, name='lien_he'),
    path('product/<slug:slug>/', views.product_detail, name='product_detail'),
    path('admin/dashboard-data/', views.dashboard_data, name='dashboard_data'),
    path('admin/dashboard/', views.dashboard, name='dashboard'),
    path('admin/', views.dashboard, name='admin_dashboard'),
    path('vouchers/', views.voucher_list, name='voucher_list'),
    path('profile/', views.user_profile, name='user_profile'),
    # New URLs for inventory forecast
    path('admin/inventory-forecast/', views.inventory_forecast, name='inventory_forecast'),
    path('admin/inventory-forecast-data/', views.inventory_forecast_data, name='inventory_forecast_data'),
    path('user-info/', views.user_info, name='user_info'),
    path('admin/order/<int:order_id>/', views.view_order, name='view_order'),
    path('admin/order/<int:order_id>/export-pdf/', views.export_order_pdf, name='export_order_pdf'),


]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)