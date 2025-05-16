# store/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import requests
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
from django.core.files.storage import default_storage
from django.conf import settings
import os
from torchvision.models import mobilenet_v2
from .forms import CreateUserForm, ReviewForm
from django.utils import timezone
from datetime import timedelta
from django.db.models import Sum, Count
from django.contrib.admin.views.decorators import staff_member_required
from decimal import Decimal
import json
from django.contrib.auth.decorators import login_required
from django.db import connection
from app.models import UserVoucher
from django.db.models.functions import Lower
from django.core.mail import send_mail
import random
import string
from .sentiment_analysis import sentiment_classifier
from .inventory_forecast import forecast_demand_and_inventory, get_inventory_forecasts
# Import các model
from .models import Category, Product, Order, OrderItem, DonHang, DonHangItem, Review, Voucher, SearchHistory
from datetime import datetime

# Initialize MobileNet V2 model
model = mobilenet_v2(weights="IMAGENET1K_V1")
model.eval()

# Setup logging
import logging
logger = logging.getLogger(__name__)
#------------------------------------------------#
# store/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
import requests
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
from django.core.files.storage import default_storage
from django.conf import settings
import os
from torchvision.models import mobilenet_v2
from .forms import CreateUserForm, ReviewForm
from django.utils import timezone
from datetime import timedelta
from django.db.models import Sum, Count
from django.contrib.admin.views.decorators import staff_member_required
from decimal import Decimal
import json
from django.contrib.auth.decorators import login_required
from django.db import connection
from app.models import UserVoucher
from django.db.models.functions import Lower
from django.core.mail import send_mail
import random
import string
from .sentiment_analysis import sentiment_classifier
from .inventory_forecast import forecast_demand_and_inventory, get_inventory_forecasts
# Import các model
from .models import Category, Product, Order, OrderItem, DonHang, DonHangItem, Review, Voucher, SearchHistory

# Initialize MobileNet V2 model
model = mobilenet_v2(weights="IMAGENET1K_V1")
model.eval()

# Setup logging
import logging
logger = logging.getLogger(__name__)

#------------------------------------------------#
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

#------------------------------------------------#
def extract_features(image):
    if isinstance(image, str):
        if not os.path.exists(image):
            logger.error(f"File does not exist: {image}")
            return None
        image = preprocess_image(image)
        if image is None:
            return None
    elif isinstance(image, torch.Tensor):
        image = image.to("cpu")
    else:
        logger.error("Invalid image input type")
        return None

    try:
        with torch.no_grad():
            features = model(image)
        return features.numpy().flatten()
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

#------------------------------------------------#
def upload_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        image_path = os.path.join(settings.MEDIA_ROOT, "uploads", image.name)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        with open(image_path, "wb") as f:
            for chunk in image.chunks():
                f.write(chunk)

        features = extract_features(image_path)
        if features is None:
            return JsonResponse({"error": "Không thể trích xuất đặc trưng từ ảnh"}, status=400)
        return JsonResponse({"features": features.tolist()})

    return JsonResponse({"error": "Không có ảnh tải lên"}, status=400)

#------------------------------------------------#
def custom_cosine_similarity(vec1, vec2):
    try:
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        norm1 = (sum(v * v for v in vec1)) ** 0.5
        norm2 = (sum(v * v for v in vec2)) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0.0

#------------------------------------------------#
def generate_product_features(product):
    """Generate and save feature vector for a product's image."""
    if not product.image or not product.image.path:
        logger.warning(f"Product {product.id} has no image")
        return None

    image_path = product.image.path
    if not os.path.exists(image_path):
        logger.error(f"Image file does not exist for product {product.id}: {image_path}")
        return None

    features = extract_features(image_path)
    if features is None:
        logger.error(f"Failed to extract features for product {product.id}")
        return None

    # Save features to a .npy file
    feature_dir = os.path.join(settings.MEDIA_ROOT, "product_features")
    os.makedirs(feature_dir, exist_ok=True)
    feature_path = os.path.join(feature_dir, f"product_{product.id}_features.npy")

    try:
        np.save(feature_path, features)
        product.feature_path = feature_path
        product.save()
        logger.info(f"Generated and saved features for product {product.id} at {feature_path}")
        return feature_path
    except Exception as e:
        logger.error(f"Error saving features for product {product.id}: {str(e)}")
        return None

#------------------------------------------------#
def search_image(request):
    if request.method == "POST" and request.FILES.get("query_image"):
        image_file = request.FILES["query_image"]
        file_path = default_storage.save("uploads/" + image_file.name, image_file)
        abs_file_path = os.path.join(settings.MEDIA_ROOT, file_path)

        image_tensor = preprocess_image(abs_file_path)
        if image_tensor is None:
            return render(request, "app/search_results.html", {
                "error": "Không thể xử lý ảnh tải lên. Vui lòng thử lại!",
                "cartItems": get_cart_items(request)
            })

        query_features = extract_features(image_tensor)
        if query_features is None:
            return render(request, "app/search_results.html", {
                "error": "Không thể trích xuất đặc trưng từ ảnh. Vui lòng thử lại!",
                "cartItems": get_cart_items(request)
            })

        # Save search history
        try:
            SearchHistory.objects.create(
                user=request.user if request.user.is_authenticated else None,
                keyword="image_search",
                product=None,
                searched_at=timezone.now()
            )
            logger.info(f"Search history saved: image_search by {request.user.username if request.user.is_authenticated else 'Anonymous'}")
        except Exception as e:
            logger.error(f"Error saving search history: {str(e)}")

        # Get products with valid feature paths
        products = Product.objects.exclude(feature_path__isnull=True).exclude(feature_path="")
        results = []

        for product in products:
            try:
                if not os.path.exists(product.feature_path):
                    logger.warning(f"Feature file missing for product {product.id}: {product.feature_path}")
                    continue

                product_features = np.load(product.feature_path)
                similarity = custom_cosine_similarity(query_features, product_features)
                results.append((product, similarity))
            except Exception as e:
                logger.error(f"Error processing features for product {product.id}: {str(e)}")
                continue

        # Sort results by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = [item[0] for item in results[:5]]  # Increased to 5 for better results
        categories = Category.objects.filter(is_sub=False)

        return render(request, "app/search_results.html", {
            "searched": "Hình ảnh",
            "matched_products": top_results,
            "categories": categories,
            "cartItems": get_cart_items(request)
        })

    return render(request, "app/search.html", {"cartItems": get_cart_items(request)})

#------------------------------------------------#
def search(request):
    searched = None
    matched_products = []

    if request.method == "POST" and "searched" in request.POST:
        searched = request.POST["searched"].strip()
        if searched:
            matched_products = Product.objects.filter(name__icontains=searched)

            # Save search history (keyword search)
            try:
                SearchHistory.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    keyword=searched,
                    product=None,
                    searched_at=timezone.now()
                )
                logger.info(f"Search history saved: {searched} by {request.user.username if request.user.is_authenticated else 'Anonymous'}")
            except Exception as e:
                logger.error(f"Error saving search history: {str(e)}")

    elif request.method == "POST" and request.FILES.get("query_image"):
        query_image = request.FILES["query_image"]
        save_dir = os.path.join(settings.MEDIA_ROOT, "query")
        os.makedirs(save_dir, exist_ok=True)

        query_image_path = os.path.join(save_dir, query_image.name)

        with open(query_image_path, "wb") as f:
            for chunk in query_image.chunks():
                f.write(chunk)

        query_features = extract_features(query_image_path)
        if query_features is None:
            return render(request, "app/search_results.html", {
                "error": "Không thể trích xuất đặc trưng từ ảnh. Vui lòng thử lại!",
                "cartItems": get_cart_items(request)
            })

        # Save search history (image search)
        try:
            SearchHistory.objects.create(
                user=request.user if request.user.is_authenticated else None,
                keyword="image_search",
                product=None,
                searched_at=timezone.now()
            )
            logger.info(f"Search history saved: image_search by {request.user.username if request.user.is_authenticated else 'Anonymous'}")
        except Exception as e:
            logger.error(f"Error saving search history: {str(e)}")

        products = Product.objects.exclude(feature_path__isnull=True).exclude(feature_path="")
        similarities = []

        for product in products:
            try:
                if not os.path.exists(product.feature_path):
                    logger.warning(f"Feature file missing for product {product.id}: {product.feature_path}")
                    continue

                product_features = np.load(product.feature_path)
                similarity = custom_cosine_similarity(query_features, product_features)
                similarities.append((product, similarity))
            except Exception as e:
                logger.error(f"Error processing features for product {product.id}: {str(e)}")
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        matched_products = [p[0] for p in similarities[:5]]  # Increased to 5 for better results

    return render(request, "app/search_results.html", {
        "searched": searched,
        "matched_products": matched_products,
        "cartItems": get_cart_items(request)
    })

#------------------------------------------------#
def about(request):
    categoryies = Category.objects.filter(is_sub=False)
    cartItems = get_cart_items(request)
    return render(request, 'app/about.html', {'categoryies': categoryies, 'cartItems': cartItems})
#------------------------------------------------#
def detail(request):
    if request.user.is_authenticated:
        customer = request.user
        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        items = []
        order = {'get_cart_items': 0, 'get_cart_total': 0}
        cartItems = order['get_cart_items']
    id = request.GET.get('id', '')
    products = Product.objects.filter(id=id)
    categoryies = Category.objects.filter(is_sub=False)
    context = {'items': items, 'order': order, 'cartItems': cartItems, 'categoryies': categoryies, 'products': products}
    return render(request, 'app/detail.html', context)
#------------------------------------------------#
def category(request):
    categoryies = Category.objects.filter(is_sub=False)
    active_category = request.GET.get('category', '')
    if active_category:
        products = Product.objects.filter(category__slug=active_category)
    else:
        products = Product.objects.all()
    cartItems = get_cart_items(request)
    context = {'categoryies': categoryies, 'products': products, 'active_category': active_category, 'cartItems': cartItems}
    return render(request, 'app/category.html', context)
#------------------------------------------------#

#------------------------------------------------#
def get_cart_items(request):
    if request.user.is_authenticated:
        order, _ = Order.objects.get_or_create(customer=request.user, complete=False)
        return order.get_cart_items
    return 0
#------------------------------------------------#
def register(request):
    form = CreateUserForm()
    if request.method == "POST":
        # Kiểm tra xem có mã xác minh được gửi chưa
        if 'verification_code' not in request.session:
            form = CreateUserForm(request.POST)
            if form.is_valid():
                # Tạo mã xác minh ngẫu nhiên (6 chữ số)
                verification_code = ''.join(random.choices(string.digits, k=6))

                # Lưu dữ liệu form và mã xác minh vào session
                request.session['temp_user_data'] = request.POST
                request.session['verification_code'] = verification_code

                # Gửi email chứa mã xác minh
                subject = 'Xác nhận đăng ký tài khoản'
                message = f'Mã xác minh của bạn là: {verification_code}\nVui lòng nhập mã này để hoàn tất đăng ký.'
                from_email = settings.DEFAULT_FROM_EMAIL
                recipient_list = [form.cleaned_data['email']]

                try:
                    send_mail(subject, message, from_email, recipient_list)
                    messages.success(request, 'Mã xác minh đã được gửi đến email của bạn. Vui lòng nhập mã để hoàn tất!')
                    return render(request, 'app/register.html', {'form': form, 'show_verification': True})
                except Exception as e:
                    messages.error(request, f'Có lỗi khi gửi email: {str(e)}')
                    return render(request, 'app/register.html', {'form': form})
        else:
            # Xử lý xác minh mã
            entered_code = request.POST.get('verification_code')
            stored_code = request.session.get('verification_code')

            if entered_code == stored_code:
                # Mã đúng, tạo tài khoản
                temp_data = request.session.get('temp_user_data')
                form = CreateUserForm(temp_data)
                if form.is_valid():
                    user = form.save()
                    messages.success(request, 'Đăng ký thành công! Vui lòng đăng nhập.')
                    # Xóa dữ liệu tạm trong session
                    del request.session['temp_user_data']
                    del request.session['verification_code']
                    return redirect('login')
            else:
                messages.error(request, 'Mã xác minh không đúng. Vui lòng thử lại!')
                return render(request, 'app/register.html', {'form': form, 'show_verification': True})

    context = {'form': form}
    return render(request, 'app/register.html', context)
#------------------------------------------------#
def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            messages.info(request, 'user or password not correct!')
    context = {}
    return render(request, 'app/login.html', context)
#------------------------------------------------#
def logoutPage(request):
    logout(request)
    return redirect('login')

#------------------------------------------------#
def home(request):
    if request.user.is_authenticated:
        customer = request.user
        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        items = []
        order = {'get_cart_items': 0, 'get_cart_total': 0}
        cartItems = order['get_cart_items']

    categoryies = Category.objects.filter(is_sub=False)
    products = Product.objects.all()

    # Log products with missing images
    for product in products:
        if not product.image:
            logger.warning(f"Product {product.name} (ID: {product.id}) has no image.")

    # ----------- XỬ LÝ GỢI Ý SẢN PHẨM DỰA TRÊN LỊCH SỬ MUA -----------
    order_items = DonHangItem.objects.all()
    product_customer_count = {}
    for item in order_items:
        product_id = item.product_id
        don_hang_id = item.don_hang_id
        if product_id not in product_customer_count:
            product_customer_count[product_id] = set()
        product_customer_count[product_id].add(don_hang_id)

    product_purchase_counts = [(pid, len(customers)) for pid, customers in product_customer_count.items()]
    product_purchase_counts.sort(key=lambda x: x[1], reverse=True)
    top_product_ids = [pid for pid, _ in product_purchase_counts[:5]]

    product_vectors = {}
    for product in products:
        vector = [
            product.price / 100,
            len(product.reviews.all()) / 5.0,
            len(product_customer_count.get(product.id, set())) / 100
        ]
        product_vectors[product.id] = vector

    product_ids = list(product_vectors.keys())
    similarity_matrix = {}
    for i, pid1 in enumerate(product_ids):
        similarity_matrix[pid1] = {}
        for j, pid2 in enumerate(product_ids):
            if i != j:
                similarity_matrix[pid1][pid2] = custom_cosine_similarity(
                    product_vectors[pid1], product_vectors[pid2]
                )

    recommended_products = Product.objects.filter(id__in=top_product_ids).order_by('-id')
    recommended_products = sorted(
        recommended_products,
        key=lambda p: len(product_customer_count.get(p.id, set())),
        reverse=True
    )

    # Log recommended products with missing images
    for product in recommended_products:
        if not product.image:
            logger.warning(f"Recommended Product {product.name} (ID: {product.id}) has no image.")

    # ----------- TÌM SẢN PHẨM ĐƯỢC TÌM KIẾM NHIỀU NHẤT -----------
    most_searched_product = None
    most_searched_keyword = SearchHistory.objects.values('keyword') \
        .annotate(search_count=Count('keyword'), normalized_keyword=Lower('keyword')) \
        .order_by('-search_count') \
        .first()

    if most_searched_keyword:
        keyword = most_searched_keyword['normalized_keyword']
        product = Product.objects.filter(name__icontains=keyword).first()
        if product:
            most_searched_product = {
                'id': product.id,
                'name': product.name,
                'price': product.price,
                'slug': product.slug,
                'image': product.image.url if product.image else '/static/app/images/default.jpg',
            }

    context = {
        'categoryies': categoryies,
        'products': products,
        'cartItems': cartItems,
        'recommended_products': recommended_products,
        'most_searched_product': most_searched_product,
    }

    return render(request, 'app/home.html', context)
#------------------------------------------------#
def cart(request):
    if request.user.is_authenticated:
        customer = request.user
        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items

        # Lấy danh sách sản phẩm đã mua với thông tin chi tiết
        order_items = DonHangItem.objects.filter(don_hang__customer=customer, don_hang__payment_status='PAID').select_related('product', 'don_hang')
        purchased_products = []
        for item in order_items:
            product_info = item.product if item.product else None
            purchased_products.append({
                'id': item.product.id if item.product else None,  # Safe access
                'name': item.ten_san_pham or (item.product.name if item.product else "Sản phẩm không xác định"),
                'image_url': item.product.ImageURL if item.product and item.product.image else '/static/app/images/default.jpg',
                'price': item.price,
                'created_at': item.don_hang.created_at
            })
        purchased_products = list({v['id']: v for v in purchased_products if v['id']}.values())  # Loại bỏ trùng lặp, chỉ giữ các item có id
    else:
        items = []
        order = {'get_cart_items': 0, 'get_cart_total': 0}
        cartItems = order['get_cart_items']
        purchased_products = []

    categoryies = Category.objects.filter(is_sub=False)
    context = {
        'items': items,
        'order': order,
        'cartItems': cartItems,
        'categoryies': categoryies,
        'purchased_products': purchased_products
    }
    return render(request, 'app/cart.html', context)
#------------------------------------------------#
def checkout(request):
    if request.method == 'POST':
        full_name = request.POST.get('name')
        email = request.POST.get('email')
        address = request.POST.get('address')
        phone = request.POST.get('mobile')
        country = request.POST.get('country')
        payment_method = request.POST.get('payment_method')
        city = request.POST.get('city')
        state = request.POST.get('state')
        voucher_code = request.POST.get('voucher_code')

        if not all([full_name, email, address, phone, country, payment_method, city, state]):
            messages.error(request, "Vui lòng điền đầy đủ thông tin giao hàng!")
            return redirect('checkout')

        if request.user.is_authenticated:
            customer = request.user
            try:
                order = Order.objects.get(customer=customer, complete=False)
                items = order.orderitem_set.all()
                if not items.exists():
                    messages.error(request, "Giỏ hàng của bạn đang trống!")
                    return redirect('cart')
                original_total = Decimal(str(order.get_cart_total))
            except Order.DoesNotExist:
                messages.error(request, "Không tìm thấy giỏ hàng!")
                return redirect('cart')
        else:
            original_total = Decimal('0.00')
            messages.error(request, "Vui lòng đăng nhập để thanh toán!")
            return redirect('login')

        # Check stock availability before proceeding
        for item in items:
            product = item.Product
            if product.stock < item.quantity:
                messages.error(request, f"Không đủ hàng trong kho cho sản phẩm {product.name}! Còn lại: {product.stock} đơn vị.")
                return redirect('cart')

        discount_applied = Decimal('0.00')
        applied_voucher = None
        total_amount = original_total
        if voucher_code:
            try:
                voucher = Voucher.objects.get(code=voucher_code)
                if UserVoucher.objects.filter(user=request.user, voucher=voucher).exists():
                    if voucher.is_valid(original_total):
                        discount_applied = voucher.discount_amount
                        total_amount = original_total - discount_applied
                        voucher.used_count += 1
                        voucher.save()
                        applied_voucher = voucher
                        messages.success(request, f"Mã voucher {voucher_code} được áp dụng thành công!")
                    else:
                        messages.error(request, "Mã voucher không hợp lệ hoặc không thể áp dụng!")
                else:
                    messages.error(request, "Vui lòng lưu voucher trước khi sử dụng!")
            except Voucher.DoesNotExist:
                messages.error(request, "Mã voucher không tồn tại!")

        if total_amount < 0:
            total_amount = Decimal('0.00')

        don_hang = DonHang.objects.create(
            customer=customer,
            full_name=full_name,
            email=email,
            address=address,
            city=city,
            state=state,
            phone=phone,
            country=country,
            total_amount=total_amount,
            payment_method=payment_method,
            payment_status='PENDING',
            discount_applied=discount_applied
        )

        if applied_voucher:
            don_hang.voucher = applied_voucher
            don_hang.save()

        for item in items:
            # Create DonHangItem
            DonHangItem.objects.create(
                don_hang=don_hang,
                product=item.Product,
                quantity=item.quantity,
                price=Decimal(str(item.Product.price)),
                ten_san_pham=item.Product.name
            )
            # Update product stock
            product = item.Product
            product.stock -= item.quantity
            if product.stock < 0:  # Safety check to prevent negative stock
                product.stock = 0
            product.save()
            logger.info(f"Updated stock for Product ID {product.id}: New stock = {product.stock}")

        order.complete = True
        order.transaction_id = f"ORDER_{order.id}_{int(order.date_order.timestamp())}"
        order.save()

        Order.objects.create(customer=customer, complete=False)

        don_hang.payment_status = 'PAID'
        don_hang.save()

        # Gửi email thông báo thanh toán
        send_payment_confirmation_email(don_hang, items, discount_applied, applied_voucher)

        cartItems = get_cart_items(request)

        return render(request, 'app/success.html', {
            'order': don_hang,
            'items': items,
            'cartItems': cartItems,
            'applied_voucher': applied_voucher,
            'discount': discount_applied
        })

    # Phần còn lại của hàm checkout (xử lý GET) giữ nguyên
    if request.user.is_authenticated:
        customer = request.user
        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        items = order.orderitem_set.all()
        cartItems = order.get_cart_items
    else:
        items = []
        order = {'get_cart_items': 0, 'get_cart_total': 0}
        cartItems = order['get_cart_items']
    categoryies = Category.objects.filter(is_sub=False)
    context = {'items': items, 'order': order, 'cartItems': cartItems, 'categoryies': categoryies}
    return render(request, 'app/checkout.html', context)
#------------------------------------------------#
def my_orders(request):
    orders = DonHang.objects.filter(customer=request.user).order_by('-created_at')
    orders_data = []
    for don_hang in orders:
        items = don_hang.items.all()
        orders_data.append({
            'don_hang': don_hang,
            'items': items
        })

    cartItems = get_cart_items(request)
    context = {
        'orders_data': orders_data,
        'cartItems': cartItems
    }
    return render(request, 'app/my_orders.html', context)
#------------------------------------------------#
@login_required
def updateItem(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Dữ liệu JSON không hợp lệ"}, status=400)

        productId = data.get("productId")
        action = data.get("action")

        if not productId or not action:
            return JsonResponse({"error": "Thiếu productId hoặc action"}, status=400)

        customer = request.user
        try:
            product = Product.objects.get(id=productId)
        except Product.DoesNotExist:
            return JsonResponse({"error": "Sản phẩm không tồn tại"}, status=404)

        order, created = Order.objects.get_or_create(customer=customer, complete=False)
        orderItem, created = OrderItem.objects.get_or_create(order=order, Product=product)

        if action == "add":
            orderItem.quantity += 1
        elif action == "remove":
            orderItem.quantity -= 1

        orderItem.save()

        if orderItem.quantity <= 0:
            orderItem.delete()

        return JsonResponse({"message": "Cập nhật thành công", "quantity": orderItem.quantity}, safe=False)

    return JsonResponse({"error": "Chỉ hỗ trợ phương thức POST"}, status=405)
#------------------------------------------------#
def lien_he(request):
    categoryies = Category.objects.filter(is_sub=False)
    cartItems = get_cart_items(request)
    return render(request, 'app/lien_he.html', {'categoryies': categoryies, 'cartItems': cartItems})
#------------------------------------------------#
def product_detail(request, slug):
    product = get_object_or_404(Product, slug=slug)
    reviews = product.reviews.all().order_by('-created_at') # Sắp xếp mới nhất lên đầu

    # --- Cập nhật SearchHistory giữ nguyên ---
    query = request.GET.get('q', '').strip()
    if query and request.user.is_authenticated:
        # Logic cập nhật SearchHistory giữ nguyên
        pass

    if request.method == 'POST':
        if not request.user.is_authenticated:
            messages.error(request, "Bạn cần đăng nhập để đánh giá sản phẩm!")
            return redirect('login')

        # Kiểm tra xem người dùng đã mua sản phẩm này chưa (Logic này cần hoàn thiện)
        # Ví dụ: đã_mua = DonHangItem.objects.filter(don_hang__customer=request.user, product=product, don_hang__payment_status='PAID').exists()
        # if not đã_mua:
        #    messages.error(request, "Bạn chỉ có thể đánh giá sản phẩm đã mua.")
        #    return redirect('product_detail', slug=slug)

        # Kiểm tra xem người dùng đã đánh giá sản phẩm này chưa
        existing_review = Review.objects.filter(product=product, user=request.user).first()
        if existing_review:
             messages.warning(request, "Bạn đã đánh giá sản phẩm này rồi.")
             form = ReviewForm(instance=existing_review) # Hiển thị lại form với đánh giá cũ nếu muốn cho sửa
             # return redirect('product_detail', slug=slug) # Hoặc redirect nếu không cho sửa
        else:
            form = ReviewForm(request.POST)
            if form.is_valid():
                review = form.save(commit=False)
                review.product = product
                review.user = request.user

                # ---- NEW: Sentiment Analysis ----
                try:
                    # Sử dụng classifier đã import
                    comment_text = review.comment
                    predicted_sentiment = sentiment_classifier.predict(comment_text)
                    review.sentiment = predicted_sentiment # Gán kết quả vào trường sentiment
                    logger.info(f"Sentiment for review (User: {review.user.username}, Product: {product.id}) : {predicted_sentiment}")
                except Exception as e:
                    # Ghi log lỗi nếu có vấn đề trong quá trình phân tích
                    logger.error(f"Sentiment analysis error for review (User: {review.user.username}, Product: {product.id}): {e}")
                    review.sentiment = None # Để trống nếu có lỗi
                # --------------------------------

                review.save() # Lưu review vào database
                messages.success(request, "Đánh giá của bạn đã được gửi và phân tích!")
                return redirect('product_detail', slug=slug)
            else:
                # NEW: Thông báo lỗi validation nếu form không hợp lệ
                messages.error(request, "Thông tin đánh giá không hợp lệ. Vui lòng kiểm tra lại.")
    else: # GET request
        if request.user.is_authenticated:
             existing_review = Review.objects.filter(product=product, user=request.user).first()
             if existing_review:
                 form = ReviewForm(instance=existing_review) # Nếu đã đánh giá, hiển thị form với dữ liệu cũ
                 messages.info(request, "Bạn đã đánh giá sản phẩm này. Bạn có thể chỉnh sửa nếu muốn.")
             else:
                 form = ReviewForm() # Form mới
        else:
            form = ReviewForm() # Form mới cho khách


    # NEW: Tính toán tóm tắt đánh giá để truyền ra template
    review_summary = product.sentiment_counts
    average_rating = product.average_rating

    context = {
        'product': product,
        'reviews': reviews,
        'form': form,
        'cartItems': get_cart_items(request),
        'review_summary': review_summary, # Truyền tóm tắt sentiment
        'average_rating': average_rating, # Truyền rating trung bình
    }

    # ---- CÁCH TEST SENTIMENT ANALYSIS TRÊN GIAO DIỆN ----
    # 1. Chạy server Django.
    # 2. Vào trang chi tiết của một sản phẩm bất kỳ.
    # 3. Đăng nhập nếu chưa đăng nhập.
    # 4. Viết một đánh giá vào form (vd: "sản phẩm tốt", "hàng kém chất lượng", "bình thường") và chọn điểm số.
    # 5. Nhấn nút "Gửi đánh giá".
    # 6. Trang sẽ tải lại. Kéo xuống phần "Đánh giá từ khách hàng".
    # 7. Xem đánh giá bạn vừa gửi. Bên cạnh thông tin đánh giá, bạn sẽ thấy một nhãn (badge) màu sắc cho biết kết quả Sentiment Analysis (Tích cực, Tiêu cực, Trung tính) dựa trên mô hình Naive Bayes đơn giản đã xử lý bình luận của bạn.
    # 8. Thử nghiệm với các bình luận khác nhau để xem kết quả dự đoán thay đổi như thế nào.
    # Lưu ý: Do mô hình đơn giản và dữ liệu huấn luyện giả lập, kết quả có thể không luôn chính xác.

    return render(request, 'app/product_detail.html', context)
#------------------------------------------------#

@staff_member_required
def dashboard(request):
    days = request.GET.get('days', 7)
    try:
        days = int(days)
    except ValueError:
        days = 7

    start_date = timezone.now() - timedelta(days=days)

    product_count = DonHangItem.objects.filter(
        don_hang__created_at__gte=start_date,
        don_hang__payment_status='PAID'
    ).aggregate(total=Sum('quantity'))['total'] or 0

    order_count = DonHang.objects.filter(
        created_at__gte=start_date,
        payment_status='PAID'
    ).count()

    total_revenue = DonHang.objects.filter(
        created_at__gte=start_date,
        payment_status='PAID'
    ).aggregate(total=Sum('total_amount'))['total'] or 0

    top_products = DonHangItem.objects.filter(
        don_hang__created_at__gte=start_date,
        don_hang__payment_status='PAID'
    ).values('ten_san_pham').annotate(sold=Sum('quantity')).order_by('-sold')[:3]

    # Lấy dữ liệu dự báo
    forecasts = get_inventory_forecasts(days=days)
    alerts = [f for f in forecasts if f['alert']]  # Thu thập các thông báo

    return render(request, 'admin/custom_dashboard.html', {
        'product_count': product_count,
        'order_count': order_count,
        'total_revenue': total_revenue,
        'top_products': [{'name': item['ten_san_pham'], 'sold': item['sold']} for item in top_products],
        'forecasts': forecasts,
        'alerts': alerts,
        'days': days,
    })

@staff_member_required
def dashboard_data(request):
    try:
        days = request.GET.get('days', 7)
        if days is None or days == '':
            return JsonResponse({
                'product_count': 0,
                'order_count': 0,
                'total_revenue': 0,
                'top_products': [],
            })

        days = int(days)
        start_date = timezone.now() - timedelta(days=days)

        product_count = DonHangItem.objects.filter(
            don_hang__created_at__gte=start_date,
            don_hang__payment_status='PAID'
        ).aggregate(total=Sum('quantity'))['total'] or 0

        order_count = DonHang.objects.filter(
            created_at__gte=start_date,
            payment_status='PAID'
        ).count()

        total_revenue = DonHang.objects.filter(
            created_at__gte=start_date,
            payment_status='PAID'
        ).aggregate(total=Sum('total_amount'))['total'] or 0

        top_products = DonHangItem.objects.filter(
            don_hang__created_at__gte=start_date,
            don_hang__payment_status='PAID'
        ).values('ten_san_pham').annotate(sold=Sum('quantity')).order_by('-sold')[:3]

        top_products_list = []
        for item in top_products:
            if 'ten_san_pham' in item and 'sold' in item:
                top_products_list.append({
                    'name': item['ten_san_pham'],
                    'sold': item['sold']
                })

        return JsonResponse({
            'product_count': product_count,
            'order_count': order_count,
            'total_revenue': float(total_revenue),
            'top_products': top_products_list,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
#------------------------------------------------#
@login_required
def voucher_list(request):
    vouchers = Voucher.objects.filter(is_active=True, valid_from__lte=timezone.now(), valid_until__gte=timezone.now())
    saved_voucher_ids = UserVoucher.objects.filter(user=request.user).values_list('voucher_id', flat=True)

    if request.method == "POST":
        voucher_id = request.POST.get('voucher_id')
        try:
            voucher = Voucher.objects.get(id=voucher_id)
            UserVoucher.objects.get_or_create(user=request.user, voucher=voucher)
            return redirect('voucher_list')
        except Voucher.DoesNotExist:
            pass

    context = {
        'vouchers': vouchers,
        'saved_voucher_ids': saved_voucher_ids,
        'cartItems': get_cart_items(request)
    }
    return render(request, 'app/voucher_list.html', context)
#------------------------------------------------#
@login_required
def user_profile(request):
    search_history = SearchHistory.objects.filter(user=request.user).order_by('-searched_at')[:10]
    context = {
        'search_history': search_history,
        'cartItems': get_cart_items(request)
    }
    return render(request, 'app/user_profile.html', context)
#------------------------------------------------#
def send_payment_confirmation_email(don_hang, items, discount_applied, applied_voucher):
    subject = 'Xác Nhận Thanh Toán Thành Công'
    item_details = "\n".join([f"- {item.Product.name} (Số lượng: {item.quantity}, Giá: {item.Product.price} VND)" for item in items])
    message = f'''
    Kính chào {don_hang.full_name},

    Cảm ơn bạn đã mua sắm tại cửa hàng của chúng tôi! Dưới đây là chi tiết đơn hàng của bạn:

    Mã đơn hàng: {don_hang.id}
    Tên khách hàng: {don_hang.full_name}
    Địa chỉ: {don_hang.address}, {don_hang.city}, {don_hang.state}, {don_hang.country}
    Số điện thoại: {don_hang.phone}
    Phương thức thanh toán: {don_hang.payment_method}
    Các sản phẩm:
    {item_details}
    Tổng tiền: {don_hang.total_amount} VND
    Giảm giá: {discount_applied} VND
    Voucher áp dụng: {applied_voucher.code if applied_voucher else "Không có"}

    Đơn hàng của bạn sẽ được xử lý sớm nhất có thể. Nếu có thắc mắc, vui lòng liên hệ qua email {settings.DEFAULT_FROM_EMAIL}.

    Trân trọng,
    Đội ngũ cửa hàng
    '''
    from_email = settings.DEFAULT_FROM_EMAIL
    recipient_list = [don_hang.email]

    try:
        send_mail(subject, message, from_email, recipient_list, fail_silently=False)
        logger.info(f"Email thông báo thanh toán đã gửi đến {don_hang.email} cho đơn hàng {don_hang.id}")
    except Exception as e:
        logger.error(f"Lỗi khi gửi email thông báo thanh toán cho đơn hàng {don_hang.id}: {str(e)}")
#------------------------------------------------#
@staff_member_required
def inventory_forecast(request):
    days = request.GET.get('days', 7)
    try:
        days = int(days)
    except ValueError:
        days = 7

    forecasts = get_inventory_forecasts(days=days)
    alerts = [f for f in forecasts if f['alert']]  # Thu thập thông báo

    # Tính toán số liệu thống kê
    total_products = len(forecasts)
    avg_current_stock = sum(f['current_stock'] for f in forecasts) / total_products if total_products > 0 else 0
    avg_predicted_demand = sum(f['predicted_demand'] for f in forecasts) / total_products if total_products > 0 else 0

    return render(request, 'admin/inventory_forecast.html', {
        'forecasts': forecasts,
        'alerts': alerts,
        'days': days,
        'total_products': total_products,
        'avg_current_stock': int(avg_current_stock),
        'avg_predicted_demand': int(avg_predicted_demand),
        'cartItems': get_cart_items(request),
    })

@staff_member_required
def inventory_forecast_data(request):
    try:
        days = request.GET.get('days', 7)
        days = int(days)
        forecasts = get_inventory_forecasts(days=days)
        alerts = [f for f in forecasts if f['alert']]
        return JsonResponse({
            'forecasts': forecasts,
            'alerts': alerts,
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
#------------------------------------------------#
def user_info(request):
    user_id = request.GET.get('userId')
    timestamp = request.GET.get('timestamp')

    # Kiểm tra timestamp hợp lệ (5 phút)
    try:
        timestamp = int(timestamp)
        current_time = int(datetime.now().timestamp() * 1000)  # Chuyển sang milliseconds
        if current_time - timestamp > 5 * 60 * 1000:  # 5 phút
            return HttpResponse('<h2>Mã QR đã hết hạn!</h2>')
    except (TypeError, ValueError):
        return HttpResponse('<h2>Mã QR không hợp lệ!</h2>')

    # Kiểm tra user_id
    if not user_id or user_id != str(request.user.id):
        return HttpResponse('<h2>Không có quyền truy cập thông tin này!</h2>')

    # Lấy thông tin người dùng
    user = request.user
    if not user.is_authenticated:
        return redirect('login')

    # Lấy danh sách đơn hàng
    orders = DonHang.objects.filter(customer=user, payment_status='PAID').order_by('-created_at')

    # Lấy danh sách sản phẩm đã mua
    order_items = DonHangItem.objects.filter(don_hang__customer=user, don_hang__payment_status='PAID').select_related('product')
    # Xử lý trường hợp product hoặc ten_san_pham có thể null
    products_bought = list(set(
        item.ten_san_pham if item.ten_san_pham else (item.product.name if item.product else "Sản phẩm không xác định")
        for item in order_items
    ))

    context = {
        'user': user,
        'orders': orders,
        'products_bought': products_bought,
    }
    return render(request, 'app/user_info.html', context)