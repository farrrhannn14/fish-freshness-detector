from ultralytics import YOLO

def main():
    model = YOLO('yolov8s.pt')

    results = model.train(
        data=f"data/data.yaml",
        epochs=50,
        imgsz=800,
        batch=16,
        patience=15,
        name='yolov8s-model',
        
        #Augmentasi
        degrees=15.0,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.3,
        mixup=0.2,
        
        #Pengaturan Belajar
        cos_lr=True,
        lr0=0.001,

        #Optimizer & Regularisasi
        optimizer='AdamW',
        weight_decay=0.0005,
        warmup_epochs=3,

        #Ekstra Tekstur & Warna
        bgr=0.1,
    )

if __name__ == '__main__':
    main()