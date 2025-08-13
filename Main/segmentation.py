import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from PIL import Image
import os

# Segmetation Model 
def load_model(model_path='../Models/Best_model.pth', num_classes=5):
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
    model = model.to('cpu')

    # strict=False로 파라미터 일부 무시하며 로드
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Segmentaion PreProcessing (Pixel값 0 ~ 1 정규화 후 Torch Tensor 변환)
def preprocess(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # [1, C, H, W]

# Segmentation Predict
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)['out']
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # [H, W]
    return pred