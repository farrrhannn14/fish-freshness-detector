import os
import shutil
from roboflow import Roboflow

def download_and_merge():
    API_KEY = "dUJo6cQS0aGViQ30r4Qt" 
    rf = Roboflow(api_key=API_KEY)
    
    # Download Dataset Utama
    project_utama = rf.workspace("ahmad-farhan-hidayat-s-workspace").project("fish-freshness-0by5o-nqtfg")
    dataset_utama = project_utama.version(3).download("yolov8")
    
    # Download Dataset Tambahan (Ikan Bergerombol)
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
                        new_tgt_file = os.path.join(tgt_path, f"{base}_new{ext}")
                        shutil.move(src_file, new_tgt_file)

    shutil.rmtree(dataset_tambahan_path, ignore_errors=True)
    final_data_dir = "data/Fish-Freshness"
    if dataset_utama_path != final_data_dir:
        shutil.move(dataset_utama_path, final_data_dir)

if __name__ == "__main__":
    download_and_merge()