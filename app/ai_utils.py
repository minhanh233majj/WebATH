# ai_utils.py
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load model (bạn có thể để ở đây để import 1 lần)
model = models.mobilenet_v2(pretrained=True)
model.classifier = torch.nn.Identity()
model.eval()

def extract_features(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy().astype('float32')
