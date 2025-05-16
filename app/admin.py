from django.contrib import admin
from django.template.response import TemplateResponse
from .models import *

# Register models
admin.site.register(Product)
admin.site.register(Category)
admin.site.register(Order)
admin.site.register(OrderItem)
admin.site.register(ShippingAddress)
admin.site.register(DonHang)
admin.site.register(Review)
admin.site.register(DonHangItem)
admin.site.register(Voucher)
admin.site.register(UserVoucher)
admin.site.register(SearchHistory)

class ReviewAdmin(admin.ModelAdmin):
    list_display = ('product', 'user', 'rating', 'comment', 'created_at')
    list_filter = ('product', 'user', 'rating', 'created_at')
    search_fields = ('product__name', 'user__username', 'comment')
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)

def custom_admin_dashboard(request):
    product_count = Product.objects.count()
    order_count = Order.objects.count()
    context = dict(
        admin.site.each_context(request),
        product_count=product_count,
        order_count=order_count,
    )
    return TemplateResponse(request, "admin/custom_dashboard.html", context)

admin.site.index = custom_admin_dashboard