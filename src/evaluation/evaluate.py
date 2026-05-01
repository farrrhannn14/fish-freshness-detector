from ultralytics import YOLO

def main():
    print("Memulai evaluasi model...")
    
    model = YOLO('models/best.pt')

    metrics = model.val(data='data/data.yaml')

    precision = metrics.box.mp
    recall = metrics.box.mr
    
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    print("\n=== HASIL EVALUASI ===")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1_score:.4f}")
    
if __name__ == '__main__':
    main()