REVISI V6 SMART NUTRISCAN AI

Fokus revisi:
1. Mengatasi halaman Oh no setelah upload gambar.
2. EasyOCR tidak lagi dimuat saat aplikasi pertama dibuka.
3. Model OCR baru dimuat ketika tombol Proses OCR ditekan.
4. Hasil OCR tidak lagi menulis langsung ke key widget Streamlit yang sudah pernah dibuat.
5. Form konfirmasi dibuat ulang dengan versi key baru setelah OCR berhasil.
6. Tampilan variasi preprocessing tidak lagi menampilkan gambar besar di expander agar memori Streamlit Cloud lebih aman.
7. Ukuran gambar OCR dan jumlah variasi preprocessing diturunkan agar risiko crash memori lebih kecil.
8. Logika klasifikasi Aman, Sedang, Tinggi dari v3 sampai v5 tetap dipertahankan.

Cara upload:
1. Ekstrak ZIP.
2. Upload semua file di dalam folder smart_nutriscan_revisi_v6 ke root repository GitHub.
3. Replace file lama.
4. Commit ke branch main.
5. Di Streamlit Cloud, pastikan Python 3.10 atau 3.11.
6. Clear cache.
7. Reboot app.

Catatan penting:
Jika masih muncul Oh no, buka Manage app, pilih Logs, lalu salin 20 sampai 40 baris error paling bawah. Tanpa log, penyebab pasti tidak bisa dilihat karena Streamlit menyembunyikan error asli di halaman publik.
