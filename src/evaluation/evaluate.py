import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

def evaluate_yolov8(model_path, data_yaml_path):
    print("\n[EVALUASI YOLOv8]")
    if not os.path.exists(model_path):
        print(f"Model YOLO tidak ditemukan: {model_path}")
        return

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml_path, split='val') 
    
    print("--- METRIKS YOLOv8 ---")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP75    : {metrics.box.map75:.4f}")

def evaluate_efficientnet(model_path, test_dir):
    print(f"\n[EVALUASI EFFICIENTNET-B3]")
    if not os.path.exists(model_path):
        print(f"Model EfficientNet tidak ditemukan: {model_path}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(os.path.join(test_dir, 'valid'), transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    class_names = test_dataset.classes

    model = models.efficientnet_b3()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n--- CLASSIFICATION REPORT EFFICIENTNET ---")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    BEST_YOLO_PATH = "models/yolo_best.pt"
    DATA_YAML = "data/Fish-Freshness/data.yaml"
    
    BEST_EFF_PATH = "models/efficientnet_best.pth"
    CLASS_DATA_DIR = "data/dataset_klasifikasi"
    
    evaluate_yolov8(BEST_YOLO_PATH, DATA_YAML)
    evaluate_efficientnet(BEST_EFF_PATH, CLASS_DATA_DIR)