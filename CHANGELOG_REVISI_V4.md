# Changelog Revisi V4

## Masalah yang ditangani

Aplikasi muncul halaman `Oh no` setelah user mengupload gambar. Ini menunjukkan error terjadi saat proses runtime, bukan saat dependency install.

## Analisis logis

Pada revisi sebelumnya OCR langsung diproses saat file gambar dipilih. Proses tersebut cukup berat karena gambar diproses dalam beberapa variasi dan EasyOCR langsung dijalankan pada rerun Streamlit. Pada Streamlit Cloud, pola ini berisiko memicu error runtime atau beban memori berlebih.

## Perubahan

1. OCR tidak otomatis berjalan setelah upload gambar.
2. Ditambahkan tombol `Proses OCR Nilai Gizi`.
3. Ditambahkan tombol `Proses OCR Komposisi`.
4. Ditambahkan wrapper `run_ocr_safely()` agar error OCR tidak menghentikan aplikasi.
5. Ditambahkan `normalize_pil_image()` untuk membatasi ukuran gambar sebelum OCR.
6. OCR memakai numpy array, bukan byte stream.
7. Parameter OCR dibuat lebih ringan.
8. Rotasi otomatis dihapus karena terlalu berat untuk Streamlit Cloud.
9. Jika OCR gagal, user tetap bisa input manual.

## File berubah

- app.py
- ocr_utils.py
- README_REVISI_V4.txt
- CHANGELOG_REVISI_V4.md

## File yang dipertahankan

- model_utils.py tetap memakai logika klasifikasi v3.
- requirements.txt tetap stabil untuk deployment.
- runtime.txt tetap berisi python 3.10.13.
