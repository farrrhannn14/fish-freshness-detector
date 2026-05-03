# Fish Freshness Detector: Deteksi Kesegaran Ikan Berbasis Computer Vision

## *Overview*
Fish Freshness Detector merupakan sistem cerdas untuk mendeteksi tingkat kesegaran ikan secara otomatis berdasarkan analisis visual pada area kulit dan mata. Sistem ini ditenagai oleh *Cascade Architecture (Two-Stage Pipeline)* yang menggabungkan kemampuan YOLOv8 dalam *object detection* dan akurasi EfficientNet-B3 dalam *image classification*. Sistem disajikan dalam bentuk RESTful API yang cepat dan responsif menggunakan FastAPI serta siap di-*deploy* dalam *environment container* (Docker).

## Arsitektur Model (*Two-Stage Pipeline*)
1. Tahap 1 (*Region Proposal*): Menggunakan YOLOv8 untuk melokalisasi dan mendeteksi titik koordinat organ spesifik ikan (mata dan kulit) dengan tingkat akurasi tinggi, bahkan pada kondisi ikan yang saling tumpang tindih (*occluded*).
2. Tahap 2 (*Freshness Classifier*): Menggunakan EfficientNet-B3 untuk memproses hasil potongan (*crop*) dari YOLOv8 dan mengekstrak detail tekstur serta degradasi warna yang selanjutnya digunakan dalam menentukan status kesegaran ikan akhir.

## Struktur Direktori
Projek ini diatur dengan struktur arsitektur berikut:
* `app/` : *Source code* untuk FastAPI (*routing*, *schema*, dan logika inferensi API).
* `models/` : Direktori penyimpanan bobot model terbaik, berisi `yolo_best.pt` dan `efficientnet_best.pth`.
* `src/` : Kumpulan *script* Python pendukung untuk dokumentasi *preprocessing*, augmentasi, pelatihan, dan evaluasi model.
* `data/` : Folder terisolasi untuk manajemen dataset lokal (diabaikan di repositori melalui `.gitignore` untuk efisiensi *storage*).
* `Dockerfile` & `.dockerignore` : Konfigurasi standar untuk *deployment container*.

## Cara Menjalankan Server (Lokal)

1. *Install* semua pustaka yang dibutuhkan (disarankan menggunakan *environment* CPU-only untuk inferensi):
```bash
pip install -r requirements.txt
```
2. Nyalakan server FastAPI:
```bash
uvicorn app.main:app --reload
```
## *Endpoint* dan Logika Inferensi
### 1. *Endpoint* `/predict`
- *Method*: `POST`
- *Body*: file (Mendukung format gambar standar: .jpg, .jpeg, .png, atau .webp)
### 2. Logika Inferensi
Sistem dirancang sedemikian rupa untuk mensimulasikan proses klasifikasi di dunia nyata:

* *On the fly cropping*: Gambar yang diunggah akan di-resize secara proporsional. Organ yang terdeteksi oleh YOLOv8 akan di-crop langsung di dalam memori (RAM) tanpa disimpan ke storage.
* Penanganan ikan bergerombol: Model telah dilatih untuk menangani foto ikan bergerombol. Jika dalam satu frame terdeteksi banyak mata atau kulit, sistem mengurutkan dan mengambil fitur dengan confidence score tertinggi sebagai acuan utama inferensi, lalu mengembalikan atribut *warning* pada respons API.
* *Pessimistic rule* (standar keamanan ketat): Sistem mengambil kesimpulan secara pesimistis demi keamanan konsumsi. Jika EfficientNet mendeteksi salah satu indikator pada fitur sebagai `Tidak Segar`, hasil `final_conclusion` akan langsung memvonis ikan tersebut sebagai Tidak Segar meskipun indikator lainnya tampak segar. Karena itu, model ini akan lebih optimal jika digunakan untuk mendeteksi ikan tunggal dalam satu foto meskipun model telah dilatih menangani ikan bergerombol.

### 3. Contoh Respons API
* Terdeteksi lebih dari satu ikan dalam foto
```JSON
{
  "filename": "ikan-kembung.webp",
  "warning": "Terdeteksi duplikasi organ tubuh ikan (lebih dari 1 mata atau kulit). Kemungkinan terdapat lebih dari 1 ikan pada gambar. Akurasi dapat menurun karena optimalisasi model hanya untuk gambar ikan tunggal.",
  "status_kesimpulan": "TIDAK SEGAR",
  "total_fitur_terdeteksi": 4,
  "detail_deteksi": [
    {
      "id_fitur": 1,
      "yolo_class": "Fresh-Skin",
      "yolo_confidence": 79.13,
      "efficientnet_prediction": "Segar",
      "efficientnet_confidence": 99.25,
      "koordinat": {
        "x1": 279,
        "y1": 40,
        "x2": 620,
        "y2": 249
      }
    },
    {
      "id_fitur": 2,
      "yolo_class": "Fresh-Skin",
      "yolo_confidence": 75.57,
      "efficientnet_prediction": "Segar",
      "efficientnet_confidence": 96.12,
      "koordinat": {
        "x1": 241,
        "y1": 174,
        "x2": 601,
        "y2": 399
      }
```
* Terdeteksi ikan tunggal
```JSON
{
  "filename": "KATALOG-MAS-768x565.jpg",
  "warning": null,
  "status_kesimpulan": "TIDAK SEGAR",
  "total_fitur_terdeteksi": 2,
  "detail_deteksi": [
    {
      "id_fitur": 1,
      "yolo_class": "Fresh-Skin",
      "yolo_confidence": 76.73,
      "efficientnet_prediction": "Segar",
      "efficientnet_confidence": 100,
      "koordinat": {
        "x1": 144,
        "y1": 188,
        "x2": 637,
        "y2": 544
      }
    },
    {
      "id_fitur": 2,
      "yolo_class": "Fresh-Eye",
      "yolo_confidence": 74.88,
      "efficientnet_prediction": "Tidak Segar",
      "efficientnet_confidence": 97.87,
      "koordinat": {
        "x1": 50,
        "y1": 353,
        "x2": 99,
        "y2": 400
      }
    }
  ]
}
```
* Tidak terdeteksi ikan
```JSON
{
  "filename": "images (1).jpeg",
  "warning": null,
  "status_kesimpulan": "TIDAK TERDETEKSI",
  "total_fitur_terdeteksi": 0,
  "detail_deteksi": []
}
```
### 4. Akses
* Live API (Hugging Face): https://frr14-fish-freshness-detector.hf.space
* Interactive Docs (Swagger UI): https://frr14-fish-freshness-detector.hf.space/docs