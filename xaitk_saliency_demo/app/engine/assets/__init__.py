from pathlib import Path
import torchvision.transforms as transforms

BASE_PATH = str(Path(__file__).parent.absolute())

with open(Path(BASE_PATH, "imagenet_classes.txt")) as f:
    imagenet_categories = f.read().splitlines()

# Pytorch pre-processing
imagenet_model_loader = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
