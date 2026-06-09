REVISI V3 SMART NUTRISCAN AI

Isi paket revisi:
1. app.py
2. model_utils.py
3. ocr_utils.py
4. requirements.txt
5. runtime.txt
6. CHANGELOG_REVISI_V3.md

Fokus perbaikan:
1. Klasifikasi Aman, Sedang, dan Tinggi dibuat konsisten dari satu fungsi pusat.
2. Data kosong atau semua nilai gizi nol tidak langsung dianalisis.
3. Rekomendasi memakai label yang sama dengan hasil klasifikasi.
4. Angka desimal pada tampilan dibatasi maksimal dua angka di belakang koma.
5. OCR tetap memakai multi preprocessing, bounding box, grouping baris, dan konfirmasi manual.
6. Batch Excel ikut memakai klasifikasi yang sama.

Cara upload:
1. Ekstrak zip.
2. Upload semua file dari folder smart_nutriscan_revisi_v3 ke root repository GitHub.
3. Replace file lama jika GitHub meminta konfirmasi.
4. Commit ke branch main.
5. Di Streamlit Cloud, pastikan Python menggunakan versi 3.10 atau 3.11.
6. Klik Clear cache.
7. Klik Reboot app.

Catatan uji:
Contoh Aman pada aplikasi akan menghasilkan klasifikasi Aman.
Contoh Sedang pada aplikasi akan menghasilkan klasifikasi Sedang.
Contoh Tinggi pada aplikasi akan menghasilkan klasifikasi Tinggi.
Data kosong akan ditahan dengan pesan Data belum cukup untuk dianalisis.
