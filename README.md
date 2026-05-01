# Fish Freshness Detector: Deteksi Kesegaran Ikan Berbasis Computer Vision

## *Overview*
Fish Freshness Detector merupakan sistem yang dirancang untuk mendeteksi tingkat kesegaran ikan berdasarkan analisis visual pada kulit dan mata ikan. Sistem ini ditenagai oleh model *Object Detection* **YOLOv8** dan disajikan dalam bentuk RESTful API responsif menggunakan FastAPI.

## Struktur Direktori
Projek ini diatur dengan struktur direktori:
* `app/` : *source code* untuk FastAPI (*routing*, *schema*, logika inferensi)
* `models/` : tempat penyimpanan *weight* model YOLOv8 terbaik (`best.pt`).
* `src/` : kumpulan *script* Python untuk dokumentasi preparasi data, pelatihan model, dan evaluasi
* `data/` : folder terisolasi untuk manajemen dataset lokal (dikosongkan di GitHub untuk menghemat memori)

## Cara Menjalankan (Server Lokal)

1. *Install* semua pustaka yang dibutuhkan:
```bash
pip install -r requirements.txt
```

2. Nyalakan server FastAPI:
```bash
uvicorn app.main:app --reload
```

3. Cek Dokumentasi API:
Buka *browser* dan kunjungi alamat berikut untuk melihat dokumentasi interaktif:
**http://127.0.0.1:8000/docs**

---

## *Endpoint* AI

### 1. *Endpoint*: `/predict`
Fungsi utama untuk mendeteksi tingkat kesegaran ikan dari sebuah unggahan foto.

* *Method*: `POST`
* *Body*: `file` (file gambar dengan format `.jpg`, `.jpeg`, `.png`, atau `.webp`)

### 2. Logika Inferensi
* Sistem mendeteksi objek dengan tingkat keyakinan (*confidence score*) minimal 65%
* Sistem hanya mendukung deteksi 1 ikan per foto. Jika ada banyak ikan yang terdeteksi, API akan memunculkan status `warning`.

### 3. Contoh Respons Sukses
Jika ikan berhasil terdeteksi sebagai ikan segar:
```json
{
  "status": "success",
  "final_conclusion": "Segar",
  "details": {
    "skin_condition": "Fresh-Skin",
    "eye_condition": "Fresh-Eye"
  },
  "raw_predictions": [
    {
      "class_name": "Fresh-Skin",
      "confidence_score": 0.829
    },
    {
      "class_name": "Fresh-Eye",
      "confidence_score": 0.838
    }
  ]
}
```

### 4. Contoh Respons Gagal (Bukan Ikan)
Jika tidak ada ikan yang terdeteksi pada gambar:
```json
{
  "status": "success",
  "final_conclusion": "Ikan Tidak Terdeteksi",
  "details": {
    "skin_condition": "Not Detected",
    "eye_condition": "Not Detected"
  },
  "raw_predictions": []
}
```