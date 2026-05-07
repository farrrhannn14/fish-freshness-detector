import os
import shutil
import cv2
from roboflow import Roboflow

API_KEY = "dUJo6cQS0aGViQ30r4Qt" 

def fix_yaml_paths(yaml_path, new_base_dir):
    if not os.path.exists(yaml_path):
        return
    with open(yaml_path, 'r') as file:
        lines = file.readlines()
    with open(yaml_path, 'w') as file:
        for line in lines:
            if line.startswith('train:'):
                file.write(f"train: {os.path.abspath(os.path.join(new_base_dir, 'train', 'images'))}\n")
            elif line.startswith('val:'):
                file.write(f"val: {os.path.abspath(os.path.join(new_base_dir, 'valid', 'images'))}\n")
            elif line.startswith('test:'):
                file.write(f"test: {os.path.abspath(os.path.join(new_base_dir, 'test', 'images'))}\n")
            else:
                file.write(line)

def crop_and_move_to_classification(yolo_dir, clf_dir):
    class_map = {
        0: 'Fresh-Eye',
        1: 'Fresh-Skin',
        2: 'NonFresh-Eye',
        3: 'NonFresh-Skin'
    }

    splits = ['train', 'valid']
    for split in splits:
        img_dir = os.path.join(yolo_dir, split, 'images')
        lbl_dir = os.path.join(yolo_dir, split, 'labels')
        
        if not os.path.exists(img_dir):
            continue
            
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + '.txt')
            
            if not os.path.exists(lbl_path): continue
                
            img = cv2.imread(img_path)
            h_img, w_img, _ = img.shape
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5: continue
                    
                class_id = int(parts[0])
                if class_id not in class_map: continue # Lewati jika bukan kelas target
                    
                class_name = class_map[class_id]
                
                x_center, y_center, w_norm, h_norm = map(float, parts[1:])
                w_box = int(w_norm * w_img)
                h_box = int(h_norm * h_img)
                x_min = int((x_center * w_img) - (w_box / 2))
                y_min = int((y_center * h_img) - (h_box / 2))
                
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w_img, x_min + w_box), min(h_img, y_min + h_box)
                
                roi = img[y_min:y_max, x_min:x_max]
                
                if roi.size == 0: continue
                
                save_folder = os.path.join(clf_dir, split, class_name)
                os.makedirs(save_folder, exist_ok=True)
                
                save_path = os.path.join(save_folder, f"crop_{idx}_{img_name}")
                cv2.imwrite(save_path, roi)
                
    print(f"Cropping selesai! Dataset klasifikasi tersimpan di: {clf_dir}")

def download_and_prepare():
    rf = Roboflow(api_key=API_KEY)
    
    # Download Dataset Utama & Tambahan
    project_utama = rf.workspace("ahmad-farhan-hidayat-s-workspace").project("fish-freshness-0by5o-nqtfg")
    dataset_utama = project_utama.version(3).download("yolov8")
    
    project_tambahan = rf.workspace("ahmad-farhan-hidayat-s-workspace").project("fish-freshness-yqy4n-ggiyd")
    dataset_tambahan = project_tambahan.version(1).download("yolov8")
    
    dataset_utama_path = dataset_utama.location
    dataset_tambahan_path = dataset_tambahan.location

    # Merge Dataset
    splits = ['train', 'valid', 'test']
    for split in splits:
        for data_type in ['images', 'labels']:
            src_path = os.path.join(dataset_tambahan_path, split, data_type)
            tgt_path = os.path.join(dataset_utama_path, split, data_type)
            
            if os.path.exists(src_path):
                os.makedirs(tgt_path, exist_ok=True)
                for filename in os.listdir(src_path):
                    src_file = os.path.join(src_path, filename)
                    tgt_file = os.path.join(tgt_path, filename)
                    if not os.path.exists(tgt_file):
                        shutil.move(src_file, tgt_file)
                    else:
                        base, ext = os.path.splitext(filename)
                        shutil.move(src_file, os.path.join(tgt_path, f"{base}_new{ext}"))

    shutil.rmtree(dataset_tambahan_path, ignore_errors=True)
    
    final_data_dir = "data/Fish-Freshness"
    if dataset_utama_path != final_data_dir:
        if os.path.exists(final_data_dir): shutil.rmtree(final_data_dir) 
        shutil.move(dataset_utama_path, final_data_dir)
        
    yaml_path = os.path.join(final_data_dir, "data.yaml")
    fix_yaml_paths(yaml_path, final_data_dir)
    
    clf_data_dir = "data/dataset_klasifikasi"
    if os.path.exists(clf_data_dir): shutil.rmtree(clf_data_dir) # Reset jika ada
    crop_and_move_to_classification(final_data_dir, clf_data_dir)

if __name__ == "__main__":
    download_and_prepare()