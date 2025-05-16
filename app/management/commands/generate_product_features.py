# app/management/commands/generate_product_features.py
from django.core.management.base import BaseCommand
from app.models import Product
from app.views import generate_product_features
import logging

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Generate and save feature vectors for all products'

    def handle(self, *args, **kwargs):
        products = Product.objects.all()
        total = len(products)
        success_count = 0
        failed_count = 0

        self.stdout.write(f"Processing {total} products...")

        for product in products:
            try:
                feature_path = generate_product_features(product)
                if feature_path:
                    success_count += 1
                    self.stdout.write(self.style.SUCCESS(f"Generated features for product {product.id}: {product.name}"))
                else:
                    failed_count += 1
                    self.stdout.write(self.style.WARNING(f"Failed to generate features for product {product.id}: {product.name}"))
            except Exception as e:
                failed_count += 1
                self.stdout.write(self.style.ERROR(f"Error processing product {product.id}: {str(e)}"))

        self.stdout.write(self.style.SUCCESS(f"Completed: {success_count} successes, {failed_count} failures"))