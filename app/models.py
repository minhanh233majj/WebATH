from django.db import models
from django.contrib.auth.models import User
import numpy as np
from django.utils.text import slugify
from django.dispatch import receiver
from django.db.models.signals import pre_save
from django.core.exceptions import ValidationError
from django.utils import timezone
from django.db.models import Sum

class Category(models.Model):
    sub_category = models.ForeignKey('self', on_delete=models.CASCADE, related_name='sub_categoryies', null=True, blank=True)
    is_sub = models.BooleanField(default=False)
    name = models.CharField(max_length=200, null=True)
    slug = models.SlugField(max_length=200, unique=True)

    def __str__(self):
        return self.name

class Product(models.Model):
    category = models.ManyToManyField('Category', related_name='product')
    name = models.CharField(max_length=200, null=True)
    price = models.FloatField()
    digital = models.BooleanField(default=False, null=True, blank=False)
    image = models.ImageField(null=True, blank=True)
    detail = models.TextField(null=True, blank=True)
    feature_path = models.CharField(max_length=500, null=True, blank=True)
    slug = models.SlugField(unique=True, blank=True, null=True)
    stock = models.IntegerField(default=0)

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug and self.name:
            self.slug = slugify(self.name)
        super(Product, self).save(*args, **kwargs)

    @property
    def ImageURL(self):
        try:
            url = self.image.url
        except:
            url = ''
        return url
    # NEW: Thêm phương thức tính rating trung bình
    @property
    def average_rating(self):
        reviews = self.reviews.all()
        if not reviews:
            return 0
        total_rating = sum(review.rating for review in reviews)
        return round(total_rating / len(reviews), 1)

    # NEW: Thêm phương thức đếm số lượng đánh giá theo cảm xúc
    @property
    def sentiment_counts(self):
        reviews = self.reviews.all()
        counts = {
            'POSITIVE': reviews.filter(sentiment='POSITIVE').count(),
            'NEGATIVE': reviews.filter(sentiment='NEGATIVE').count(),
            'NEUTRAL': reviews.filter(sentiment='NEUTRAL').count(),
            'TOTAL': reviews.count()
        }
        return counts

class Order(models.Model):
    customer = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True)
    date_order = models.DateTimeField(auto_now_add=True)
    complete = models.BooleanField(default=False, null=True, blank=False)
    transaction_id = models.CharField(max_length=200, null=True)

    def __str__(self):
        return str(self.id)

    @property
    def get_cart_items(self):
        orderitems = self.orderitem_set.all()
        total = sum([item.quantity for item in orderitems])
        return total

    @property
    def get_cart_total(self):
        orderitems = self.orderitem_set.all()
        total = sum([item.get_total for item in orderitems])
        return float(total)

class OrderItem(models.Model):
    Product = models.ForeignKey(Product, on_delete=models.SET_NULL, blank=True, null=True)
    order = models.ForeignKey(Order, on_delete=models.SET_NULL, blank=True, null=True)
    quantity = models.IntegerField(default=0, null=True, blank=True)
    date_added = models.DateTimeField(auto_now_add=True)

    @property
    def get_total(self):
        total = self.Product.price * self.quantity if self.Product and self.quantity else 0
        return total

class ShippingAddress(models.Model):
    customer = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True)
    order = models.ForeignKey(Order, on_delete=models.SET_NULL, blank=True, null=True)
    address = models.CharField(max_length=200, null=True)
    city = models.CharField(max_length=200, null=True)
    state = models.CharField(max_length=200, null=True)
    mobile = models.CharField(max_length=10, null=True)
    date_added = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.address

class DonHang(models.Model):
    PAYMENT_METHODS = (
        ('COD', 'Thanh toán khi nhận hàng'),
        ('VNPAY', 'VNPay'),
    )
    PAYMENT_STATUSES = (
        ('PENDING', 'Pending'),
        ('PAID', 'Paid'),
        ('FAILED', 'Failed'),
    )
    customer = models.ForeignKey(User, on_delete=models.SET_NULL, blank=True, null=True)
    full_name = models.CharField(max_length=100)
    email = models.EmailField()
    address = models.CharField(max_length=255)
    city = models.CharField(max_length=100, null=True, blank=True)
    state = models.CharField(max_length=100, null=True, blank=True)
    phone = models.CharField(max_length=20)
    country = models.CharField(max_length=100)
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    payment_method = models.CharField(max_length=10, choices=PAYMENT_METHODS)
    payment_status = models.CharField(max_length=20, choices=PAYMENT_STATUSES, default='PENDING')
    created_at = models.DateTimeField(auto_now_add=True)
    discount_applied = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    voucher = models.ForeignKey('Voucher', on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"DonHang #{self.id} - {self.full_name}"

    def get_original_total(self):
        return sum(item.get_total() for item in self.items.all()) + self.discount_applied

class DonHangItem(models.Model):
    don_hang = models.ForeignKey('DonHang', on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey('Product', on_delete=models.SET_NULL, null=True, blank=True)
    quantity = models.IntegerField(default=1)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    ten_san_pham = models.CharField(max_length=200, blank=True, null=True)

    def get_total(self):
        return self.quantity * self.price if self.quantity and self.price else 0

    def __str__(self):
        return f"{self.quantity} x {self.ten_san_pham} in DonHang {self.don_hang.id}" if self.ten_san_pham else f"{self.quantity} x Unknown Product in DonHang {self.don_hang.id}"

@receiver(pre_save, sender=DonHangItem)
def set_ten_san_pham(sender, instance, **kwargs):
    if instance.product:
        instance.ten_san_pham = instance.product.name
    else:
        instance.ten_san_pham = None

class Review(models.Model):
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    # NEW: Thêm trường sentiment để lưu kết quả phân tích
    sentiment = models.CharField(
        max_length=10,
        choices=[('POSITIVE', 'Tích cực'), ('NEGATIVE', 'Tiêu cực'), ('NEUTRAL', 'Trung tính')],
        null=True, blank=True # Cho phép null ban đầu hoặc nếu phân tích lỗi
    )

    def __str__(self):
        return f"Review by {self.user.username} for {self.product.name} ({self.sentiment or 'Chưa phân tích'})"


class Voucher(models.Model):
    code = models.CharField(max_length=50, unique=True)
    discount_amount = models.DecimalField(max_digits=10, decimal_places=2)
    min_order_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    valid_from = models.DateTimeField()
    valid_until = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    max_usage = models.IntegerField(default=1)
    used_count = models.IntegerField(default=0)

    def clean(self):
        if self.valid_until < self.valid_from:
            raise ValidationError("Ngày hết hạn phải sau ngày bắt đầu.")
        if self.discount_amount <= 0:
            raise ValidationError("Số tiền giảm giá phải lớn hơn 0.")
        if self.min_order_amount < 0:
            raise ValidationError("Số tiền đơn hàng tối thiểu không được âm.")
        if self.max_usage <= 0:
            raise ValidationError("Số lần sử dụng tối đa phải lớn hơn 0.")

    def is_valid(self, order_total, user=None):
        now = timezone.now()
        user_usage = 0
        if user:
            user_voucher_usage = UserVoucher.objects.filter(user=user, voucher=self).first()
            if user_voucher_usage:
                user_usage = user_voucher_usage.used_count # Sửa lại cách lấy used_count
            else:
                user_usage = 0 # Nếu user chưa từng lưu voucher này


        is_voucher_valid = (
            self.is_active and
            self.valid_from <= now <= self.valid_until and
            self.used_count < self.max_usage and
            order_total >= self.min_order_amount
        )

        is_user_eligible = True
        if user:
             # Kiểm tra xem user đã sử dụng voucher này qua bảng UserVoucher chưa
             is_user_eligible = not UserVoucher.objects.filter(user=user, voucher=self, used_count__gte=1).exists()


        return is_voucher_valid and is_user_eligible


    def __str__(self):
        return self.code

class UserVoucher(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='saved_vouchers')
    voucher = models.ForeignKey('Voucher', on_delete=models.CASCADE)
    saved_at = models.DateTimeField(auto_now_add=True)
    used_count = models.IntegerField(default=0) # Thêm trường này để theo dõi số lần user đã dùng

    class Meta:
        unique_together = ('user', 'voucher')

    def __str__(self):
        return f"{self.user.username} - {self.voucher.code} (Used: {self.used_count})"

class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='search_history', null=True, blank=True)
    keyword = models.CharField(max_length=200)
    product = models.ForeignKey('Product', on_delete=models.SET_NULL, null=True, blank=True, related_name='search_history')
    searched_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.user.username if self.user else 'Anonymous'} - {self.keyword} - {self.searched_at}"

    class Meta:
        ordering = ['-searched_at']
        indexes = [
            models.Index(fields=['user', 'searched_at']),
            models.Index(fields=['keyword']),
        ]
