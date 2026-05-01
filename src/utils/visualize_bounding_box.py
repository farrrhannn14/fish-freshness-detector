import cv2
from ultralytics import YOLO

def draw_predictions(image_path, model_path="models/best.pt", output_path="hasil_visualisasi.jpg"):
    model = YOLO(model_path)
    
    results = model.predict(image_path, conf=0.65)
    
    annotated_frame = results[0].plot()
    
    cv2.imwrite(output_path, annotated_frame)
    print(f"Gambar dengan bounding box berhasil disimpan di: {output_path}")

if __name__ == "__main__":
    gambar_test = "data/raw/sample_ikan_tidak_segar.jpeg"
    draw_predictions(gambar_test)