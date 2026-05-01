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

---

## *Endpoint* AI

### 1. *Endpoint*: `/predict`
Fungsi utama untuk mendeteksi tingkat kesegaran ikan dari sebuah unggahan foto.

* *Method*: `POST`
* *Body*: `file` (file gambar dengan format `.jpg`, `.jpeg`, `.png`, atau `.webp`)

### 2. Logika Inferensi
Sistem klasifikasi ini menggunakan ekstraksi fitur spesifik (mata dan kulit) dengan alur logika sebagai berikut:

* **Threshold & Resolusi:** Model melakukan prediksi dengan ambang batas keyakinan (*confidence score*) sebesar 45.8% (sesuai evaluasi F1 dan *confidence score*) dan gambar akan diubah ukurannya (*resize*) secara internal menjadi 800px untuk mempertahankan detail fitur-fitur kecil.
* **Resolusi Konflik Multifitur:** Apabila dalam satu *frame* terdeteksi lebih dari satu area mata atau kulit (misalnya karena ada tumpukan ikan), sistem akan secara otomatis mengambil fitur dengan tingkat keyakinan (*confidence*) paling tinggi sebagai acuan utama, dan memunculkan pesan peringatan di respons API.
* **Logika Kesimpulan (*Pessimistic Rule*):** Sistem menerapkan standar keamanan kualitas yang ketat. Jika salah satu indikator (baik itu kulit maupun mata) terdeteksi sebagai `NonFresh`, hasil akhir (`final_conclusion`) akan langsung dikategorikan sebagai `Tidak Segar`. Ikan hanya dinyatakan `Segar` jika indikator yang terdeteksi mengarah pada kelas segar.

### 3. Contoh Respons Sukses
Jika ikan berhasil terdeteksi sebagai ikan segar:
```json
{
  "status": "success",
  "message": "Analisis kesegaran ikan berhasil dilakukan.",
  "final_conclusion": "Segar",
  "details": {
    "skin_condition": "Fresh-Skin",
    "eye_condition": "Fresh-Eye",
    "skin_count": 1,
    "eye_count": 1
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
  "message": "Pastikan gambar yang diunggah adalah gambar ikan yang jelas.",
  "final_conclusion": "Ikan Tidak Terdeteksi",
  "details": {
    "skin_condition": "Not Detected",
    "eye_condition": "Not Detected",
    "skin_count": 0,
    "eye_count": 0
  },
  "raw_predictions": []
}
```

## 🔗 Link Akses
* **Live API (Hugging Face):** [https://huggingface.co/spaces/frr14/fish-freshness-detector](https://frr14-fish-freshness-detector.hf.space)
* **Interactive Docs (Swagger UI):** [https://frr14-fish-freshness-detector.hf.space/docs](https://frr14-fish-freshness-detector.hf.space/docs)
