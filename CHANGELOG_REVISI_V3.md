# Changelog Revisi V3 SMART NutriScan AI

## Perbaikan utama

1. Menambahkan fungsi `classify_risk()` di `model_utils.py` sebagai satu sumber klasifikasi.
2. Menambahkan fungsi `has_sufficient_input()` agar data kosong tidak diberi hasil risiko palsu.
3. Mengubah `analyze_product_fully()` agar memakai skor gizi terkalibrasi sebagai pengaman ketika model ML menyimpang ekstrem.
4. Memperbaiki pembacaan probabilitas model binary dan multiclass.
5. Menyamakan label hasil analisis dan rekomendasi.
6. Membatasi seluruh tampilan angka utama menjadi dua angka desimal.
7. Menambahkan contoh uji Aman, Sedang, dan Tinggi sesuai dokumen revisi.
8. Menambahkan validasi batch Excel untuk data kosong.
9. Mempertahankan OCR multi preprocessing di `ocr_utils.py`.
10. Mempertahankan `safe_image()` agar tetap kompatibel dengan versi Streamlit lama dan baru.

## File yang berubah

1. `app.py`
2. `model_utils.py`
3. `ocr_utils.py`
4. `requirements.txt`
5. `runtime.txt`

## Prinsip revisi

Aplikasi tidak boleh memberi klasifikasi tinggi ketika data nutrisi kosong. OCR hanya dipakai untuk mengisi data awal. Pengguna tetap harus mengonfirmasi data sebelum analisis dijalankan.
