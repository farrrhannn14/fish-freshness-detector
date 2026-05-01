import os
from roboflow import Roboflow
from PIL import Image

def download_dataset():
    rf = Roboflow(api_key="dUJo6cQS0aGViQ30r4Qt")
    project = rf.workspace("ahmad-farhan-hidayat-s-workspace").project("fish-freshness-0by5o-nqtfg")
    version = project.version(3)
    dataset = version.download("yolov8")

def verify_images(base_dir):
    corrupted_images = 0
    total_images = 0

    for root, dirs, files in os.walk(base_dir):
        if 'images' in root:
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    total_images += 1
                    file_path = os.path.join(root, file)
                    try:
                        img = Image.open(file_path)
                        img.verify()
                    except (IOError, SyntaxError) as e:
                        print(f"Gambar Cacat Terdeteksi: {file_path}")
                        corrupted_images += 1

    print(f"Hasil Pengecekan:")
    print(f"- Total gambar diperiksa : {total_images}")
    print(f"- Gambar cacat     : {corrupted_images}")
    
    if corrupted_images == 0:
        print("Dataset sehat dan siap untuk digunakan")
    else:
        print("Ada gambar yang cacat!")

def main():
    #download_dataset()
    dataset_path = "data/train"
    if os.path.exists(dataset_path):
        verify_images(dataset_path)
    else:
        print("Folder dataset belum ada, silakan jalankan fungsi download_dataset() terlebih dahulu.")

if __name__ == '__main__':
    main()