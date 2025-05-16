import math
from django.utils import timezone
from datetime import timedelta
from .models import DonHangItem, Product
from django.db.models import Sum
import logging

logger = logging.getLogger(__name__)

class LinearRegression:
    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0

    def fit(self, X, y):
        n = len(X)
        if n == 0 or len(y) != n:
            logger.warning("Không đủ dữ liệu để huấn luyện mô hình hồi quy tuyến tính.")
            return

        mean_x = sum(X) / n
        mean_y = sum(y) / n

        numerator = 0
        denominator = 0
        for i in range(n):
            numerator += (X[i] - mean_x) * (y[i] - mean_y)
            denominator += (X[i] - mean_x) ** 2

        if denominator != 0:
            self.slope = numerator / denominator
            self.intercept = mean_y - self.slope * mean_x
        else:
            logger.warning("Độ dốc không thể tính do mẫu số bằng 0. Đặt slope và intercept về 0.")
            self.slope = 0
            self.intercept = mean_y if n > 0 else 0

    def predict(self, X):
        predictions = [self.slope * x + self.intercept for x in X]
        return [max(0, int(round(pred))) for pred in predictions]  # Đảm bảo giá trị dự đoán không âm

def forecast_demand_and_inventory(product_id, days=7, lead_time_days=3, safety_stock_factor=0.2):
    """
    Dự báo nhu cầu và tính toán nhu cầu nhập kho cho một sản phẩm.
    """
    # Lấy dữ liệu lịch sử bán hàng
    end_date = timezone.now()
    start_date = end_date - timedelta(days=days)
    sales = DonHangItem.objects.filter(
        product_id=product_id,
        don_hang__created_at__gte=start_date,
        don_hang__created_at__lte=end_date,
        don_hang__payment_status='PAID'
    ).values('don_hang__created_at__date').annotate(total_quantity=Sum('quantity')).order_by('don_hang__created_at__date')

    # Lấy thông tin sản phẩm trước
    try:
        product = Product.objects.get(id=product_id)
        current_stock = product.stock
        product_name = product.name
        logger.info(f"Sản phẩm ID {product_id} - Tên: {product_name}, Tồn kho: {current_stock}")
    except Product.DoesNotExist as e:
        logger.error(f"Sản phẩm với ID {product_id} không tồn tại: {e}")
        current_stock = 0
        product_name = 'Unknown Product'

    # Kiểm tra dữ liệu lịch sử
    if not sales:
        logger.warning(f"Không có dữ liệu bán hàng cho sản phẩm ID {product_id}")
        return {
            'product_id': product_id,
            'product_name': product_name,
            'predicted_demand': 0,
            'current_stock': current_stock,
            'restock_quantity': 0,
            'alert': f'Không có dữ liệu bán hàng (ID: {product_id})'
        }

    # Chuẩn bị dữ liệu cho hồi quy tuyến tính
    X = []
    y = []
    base_date = start_date.date()
    for sale in sales:
        day = (sale['don_hang__created_at__date'] - base_date).days
        X.append(day)
        y.append(sale['total_quantity'])

    # Kiểm tra số ngày có dữ liệu bán hàng
    unique_days = len(set(X))
    avg_demand_per_day = sum(y) / unique_days if unique_days > 0 else 0

    # Nếu chỉ có dữ liệu trong 1 ngày, không dùng hồi quy mà lấy trung bình
    if unique_days <= 1:
        logger.info(f"Sản phẩm ID {product_id} chỉ có dữ liệu trong {unique_days} ngày. Sử dụng trung bình lịch sử.")
        predicted_demand = int(round(avg_demand_per_day * lead_time_days))
    else:
        # Huấn luyện mô hình hồi quy tuyến tính
        model = LinearRegression()
        model.fit(X, y)

        # Dự đoán nhu cầu cho lead_time_days tiếp theo
        future_days = [max(X) + i + 1 for i in range(lead_time_days)]
        predicted_values = model.predict(future_days)
        predicted_demand = sum(predicted_values)

        # Nếu hồi quy trả về 0, sử dụng trung bình lịch sử làm dự đoán
        if predicted_demand == 0 and y:
            predicted_demand = int(round(avg_demand_per_day * lead_time_days))
            logger.info(f"Dự đoán bằng hồi quy trả về 0. Sử dụng trung bình lịch sử: {predicted_demand}")

    # Tính toán tồn kho an toàn và số lượng cần nhập
    safety_stock = predicted_demand * safety_stock_factor
    restock_quantity = max(0, predicted_demand + safety_stock - current_stock)

    # Sinh thông báo
    alert = None
    if current_stock < predicted_demand:
        alert = f"Tồn kho thấp! Tồn kho hiện tại ({current_stock}) nhỏ hơn nhu cầu dự đoán ({predicted_demand})."
    elif predicted_demand > avg_demand_per_day * 1.5 and predicted_demand > 0:  # Nếu nhu cầu dự đoán cao hơn 50% so với trung bình
        alert = f"Nhu cầu cao dự đoán ({predicted_demand}) cho sản phẩm ID {product_id}."

    return {
        'product_id': product_id,
        'product_name': product_name,
        'predicted_demand': int(predicted_demand),
        'current_stock': current_stock,
        'restock_quantity': int(restock_quantity),
        'alert': alert
    }

def get_inventory_forecasts(days=7):
    """
    Tạo dự báo cho tất cả sản phẩm.
    """
    products = Product.objects.all()
    forecasts = []
    for product in products:
        forecast = forecast_demand_and_inventory(product.id, days=days)
        forecasts.append(forecast)
    logger.info(f"Dự báo đã tạo: {forecasts}")
    return forecasts
