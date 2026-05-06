import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = {
    0: "Fresh-Eye", 
    1: "Fresh-Skin", 
    2: "NonFresh-Eye", 
    3: "NonFresh-Skin"
}

def load_yolo_labels(label_dir):
    all_data = []
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    for file in txt_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    bbox_area = width * height
                    aspect_ratio = width / height if height > 0 else 0
                    
                    all_data.append([
                        CLASS_NAMES.get(class_id, f"Class {class_id}"), 
                        x_center, y_center, width, height, bbox_area, aspect_ratio
                    ])
                    
    return pd.DataFrame(all_data, columns=["Class", "X_Center", "Y_Center", "Width", "Height", "Bbox_Area", "Aspect_Ratio"])

def plot_bbox_distribution(label_dir, output_filename="data/distribusi_bounding_box.png"):
    print(f"Mencari file label di: {label_dir}...")
    df_labels = load_yolo_labels(label_dir)
    print(f"Total Bounding Box ditemukan: {len(df_labels)}")
    
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df_labels, x="Class", order=df_labels["Class"].value_counts().index)
    
    plt.title("Distribusi Jumlah Bounding Box per Kelas", fontsize=14, fontweight='bold')
    plt.ylabel("Jumlah Anotasi")
    plt.xlabel("Kategori Kelas")
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points')
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot berhasil disimpan sebagai '{output_filename}'")

if __name__ == "__main__":
    LABEL_DIR = "data/Fish-Freshness/train/labels"
    if os.path.exists(LABEL_DIR):
        plot_bbox_distribution(LABEL_DIR)
    else:
        print(f"Folder {LABEL_DIR} tidak ditemukan. Jalankan prepare_data.py terlebih dahulu.")