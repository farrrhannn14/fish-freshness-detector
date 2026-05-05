import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from ultralytics import YOLO

# STAGE 1: YOLOv8 TRAINING
def train_yolov8(data_yaml_path):
    model = YOLO('yolov8s.pt') 
    
    results = model.train(
        data=data_yaml_path,
        epochs=50,  
        imgsz=800,
        batch=16,
        patience=15,
        name='yolov8s-model',
        degrees=15.0,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.3,
        mixup=0.2,
        cos_lr=True,
        lr0=0.001,
        optimizer='AdamW',
        weight_decay=0.0005,
        warmup_epochs=3
    )
    print("Training YOLOv8 Selesai!\n")
    return results

# STAGE 2: EFFICIENTNET-B3 TRAINING
def train_efficientnet(dataset_dir, num_epochs=10, batch_size=16):
    print(f"\n[STAGE 2] Memulai Training EfficientNet-B3 dari {dataset_dir}...")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x), data_transforms[x]) 
                      for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) 
                   for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes

    print(f"Kelas ditemukan: {class_names}")

    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'Training selesai. Best Valid Acc: {best_acc:4f}')
    
    os.makedirs('models', exist_ok=True)
    save_path = 'models/efficientnet_best.pth'
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), save_path)
    print(f"Model EfficientNet disimpan di: {save_path}")

if __name__ == "__main__":
    YOLO_DATA_YAML = "data/Fish-Freshness/data.yaml"
    if os.path.exists(YOLO_DATA_YAML):
        train_yolov8(YOLO_DATA_YAML)
        
    EFFICIENTNET_DATA_DIR = "data/dataset_klasifikasi"
    if os.path.exists(EFFICIENTNET_DATA_DIR):
        train_efficientnet(EFFICIENTNET_DATA_DIR, num_epochs=10)