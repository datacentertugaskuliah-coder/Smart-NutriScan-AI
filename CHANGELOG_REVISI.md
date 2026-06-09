# Changelog Revisi Smart NutriScan AI

## Fokus perbaikan

Revisi ini memperbaiki dua masalah utama.

Pertama, deployment Streamlit gagal karena Python cloud terbaca sebagai Python 3.14.5, sementara TensorFlow dan beberapa package numerik belum cocok dengan versi tersebut.

Kedua, OCR lama belum stabil karena hanya memakai satu preprocessing, membaca teks tanpa posisi, dan parsing nilai gizi dari gabungan semua teks. Revisi ini mengubah OCR menjadi berbasis beberapa preprocessing, detail bounding box, grouping baris, dan parsing berbasis baris.

## File yang diubah

1. `requirements.txt`
2. `runtime.txt`
3. `app.py`
4. `model_utils.py`
5. `ocr_utils.py`

## Prinsip integritas

Aplikasi tidak langsung membuat rekomendasi dari hasil OCR mentah. OCR hanya mengisi data awal. Pengguna tetap harus mengecek dan mengoreksi hasil pada form konfirmasi sebelum analisis dijalankan.
