import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class_names = [
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
]
num_classes = len(class_names)



# DenseNet-201
#MODEL_TYPE = "densenet201"
#MODEL_PATH = "test_bestptler/best_densenet201_focal_adam.pth"
# best_densenet201_focal_adamw.pth
# best_densenet201_ce_adamw.pth
#


# DenseNet-121
MODEL_TYPE = "densenet121"
MODEL_PATH = "test_bestptler/best_densenet121_parallel.pth"


# EfficientNetV2-S 
#MODEL_TYPE = "efficientnetv2"
#MODEL_PATH = "test_bestptler/best_garbage_model_regularized.pth"
# best_garbage_model_adam_ce.pth
# best_garbage_model_adam_ce_dropout.pth
#


def load_densenet121(model_path):
    model = models.densenet121(weights=None)

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(model.classifier.in_features),
        nn.Dropout(p=0.5),
        nn.Linear(model.classifier.in_features, num_classes)
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model

def load_densenet201(model_path):
    model = models.densenet201(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def load_efficientnetv2(model_path):
    model = models.efficientnet_v2_s(weights=None)

    # 🔥 EĞİTİMDE KULLANDIĞIN CLASSIFIER YAPISI (AYNI!)
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(1280),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
    )

    state_dict = torch.load(model_path, map_location=device)

    # Eğer DataParallel ile eğitildiyse (olabilir)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if MODEL_TYPE == "densenet121":
    model = load_densenet121(MODEL_PATH)
elif MODEL_TYPE == "densenet201":
    model = load_densenet201(MODEL_PATH)
elif MODEL_TYPE == "efficientnetv2":
    model = load_efficientnetv2(MODEL_PATH)
else:
    raise ValueError("Geçersiz MODEL_TYPE")

print(f"Model yüklendi: {MODEL_TYPE}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


def predict_image(model, image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return class_names[pred.item()], conf.item()


IMAGE_PATH = "test_bestptler/istockphoto-465864920-612x612.jpg"


predicted_class, confidence = predict_image(model, IMAGE_PATH)


print(f" Kullanılan Model : {MODEL_TYPE}")
print(f" Model Dosyası   : {os.path.basename(MODEL_PATH)}")
print(f" Tahmin          : {predicted_class}")
print(f" Güven Skoru     : %{confidence * 100:.2f}")
