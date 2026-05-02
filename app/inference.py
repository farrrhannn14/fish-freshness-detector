import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

YOLO_MODEL_PATH = "models/yolo_best.pt"
EFFNET_MODEL_PATH = "models/efficientnet_best.pth"

CLASS_NAMES = ['Segar', 'Tidak_Segar']

model_yolo = YOLO(YOLO_MODEL_PATH)
device = torch.device("cpu")

model_ft = models.efficientnet_b3(weights=None)
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))

model_ft.load_state_dict(torch.load(EFFNET_MODEL_PATH, map_location=device))
model_ft.eval()

img_transforms = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_inference(image_bytes: bytes, filename: str) -> dict:
    try:
        original_img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Gagal membaca gambar: {str(e)}")

    results = model_yolo.predict(source=original_img_pil, conf=0.417, save=False, verbose=False)
    result = results[0]
    
    is_non_fresh = False
    details = []

    warning_msg = None
    
    if len(result.boxes) > 0:
        base_features = []
        for box in result.boxes:
            raw_class_name = model_yolo.names[int(box.cls[0])].lower()
            
            base_organ = (raw_class_name
                          .replace("nonfresh", "")
                          .replace("fresh", "")
                          .replace("tidak_segar", "")
                          .replace("segar", "")
                          .replace("_", "")
                          .strip())
            
            base_features.append(base_organ)
        
        if len(base_features) != len(set(base_features)):
            warning_msg = "Terdeteksi duplikasi organ tubuh ikan (lebih dari 1 mata atau kulit). Kemungkinan terdapat lebih dari 1 ikan pada gambar. Akurasi dapat menurun karena optimalisasi model hanya untuk gambar ikan tunggal."

        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            yolo_class_name = model_yolo.names[class_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            cropped_img = original_img_pil.crop((x1, y1, x2, y2))
            input_tensor = img_transforms(cropped_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model_ft(input_tensor)
                
                probabilities = F.softmax(outputs, dim=1)
                eff_conf_tensor, preds = torch.max(probabilities, 1)
                eff_conf = float(eff_conf_tensor[0].item())
                eff_class = CLASS_NAMES[preds[0].item()]
            
            if eff_class == "Tidak_Segar" and eff_conf > 0.65:
                is_non_fresh = True
            elif eff_class == "Tidak_Segar" and eff_conf <= 0.65:
                eff_class = "Tidak_Segar (Dianulir karena keraguan)"
                
            details.append({
                "id_fitur": i + 1,
                "yolo_class": yolo_class_name,
                "yolo_confidence": round(conf * 100, 2),
                "efficientnet_prediction": eff_class.replace('_', ' '),
                "efficientnet_confidence": round(eff_conf * 100, 2),
                "koordinat": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })

    if len(result.boxes) == 0:
        final_conclusion = "TIDAK TERDETEKSI"
    else:
        final_conclusion = "TIDAK SEGAR" if is_non_fresh else "SEGAR"

    return {
        "filename": filename,
        "warning": warning_msg,
        "status_kesimpulan": final_conclusion,
        "total_fitur_terdeteksi": len(result.boxes),
        "detail_deteksi": details
    }