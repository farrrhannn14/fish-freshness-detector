from ultralytics import YOLO
from PIL import Image
import io

model = YOLO("models/best.pt")

def predict_fish(image_bytes):
    try:
        raw_image = Image.open(io.BytesIO(image_bytes))
        
        if raw_image.mode in ('RGBA', 'LA') or (raw_image.mode == 'P' and 'transparency' in raw_image.info):
            alpha = raw_image.convert('RGBA').split()[-1]
            image = Image.new("RGB", raw_image.size, (255, 255, 255))
            image.paste(raw_image, mask=alpha)
        else:
            image = raw_image.convert("RGB")
        
    except Exception as e:
        return {"status": "error", "message": f"Gagal membaca gambar: {str(e)}"}
    
    results = model.predict(image, conf=0.458, imgsz=800)
    
    predictions = []
    skin_detections = []
    eye_detections = []
    skin_count = 0
    eye_count = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]
            
            predictions.append({
                "class_name": class_name,
                "confidence_score": round(confidence, 3)
            })
            
            if "Skin" in class_name:
                skin_detections.append((class_name, confidence))
                skin_count += 1
            elif "Eye" in class_name:
                eye_detections.append((class_name, confidence))
                eye_count += 1
        
    skin_status = "Not Detected"
    eye_status = "Not Detected"
    
    if skin_detections:
        skin_status = max(skin_detections, key=lambda x: x[1])[0]
    if eye_detections:
        eye_status = max(eye_detections, key=lambda x: x[1])[0]

    final_conclusion = "Unknown"
    if skin_status == "Not Detected" and eye_status == "Not Detected":
        final_conclusion = "Ikan Tidak Terdeteksi"
    elif skin_status == "NonFresh-Skin" or eye_status == "NonFresh-Eye":
        final_conclusion = "Tidak Segar"
    else:
        final_conclusion = "Segar"

    if skin_count > 1 or eye_count > 1:
        info_message = f"{skin_count} area kulit dan {eye_count} area mata terdeteksi. Kesimpulan diambil berdasarkan fitur yang paling jelas."
    elif skin_count == 0 and eye_count == 0:
        info_message = "Pastikan gambar yang diunggah adalah gambar ikan yang jelas."
    else:
        info_message = "Analisis kesegaran ikan berhasil dilakukan."
        
    return {
        "status": "success",
        "message": info_message,
        "final_conclusion": final_conclusion,
        "details": {
            "skin_condition": skin_status,
            "eye_condition": eye_status,
            "skin_count": skin_count,
            "eye_count": eye_count
        },
        "raw_predictions": predictions
    }