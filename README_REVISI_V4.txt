REVISI V4 SMART NUTRISCAN AI

Fokus revisi:
1. Mencegah aplikasi langsung crash saat gambar diupload.
2. OCR tidak lagi berjalan otomatis saat file dipilih.
3. User harus menekan tombol Proses OCR Nilai Gizi atau Proses OCR Komposisi.
4. Ukuran gambar dibatasi sebelum OCR agar aman di Streamlit Cloud.
5. EasyOCR dibuat lebih ringan dan lebih kompatibel.
6. Jika OCR gagal, aplikasi menampilkan pesan error di halaman, bukan langsung Oh no.
7. Logika klasifikasi v3 tetap dipertahankan.

Cara upload:
1. Ekstrak ZIP.
2. Upload semua file di dalam folder smart_nutriscan_revisi_v4 ke root repository GitHub.
3. Replace file lama.
4. Commit ke branch main.
5. Di Streamlit Cloud, pastikan Python 3.10 atau 3.11.
6. Clear cache.
7. Reboot app.

Catatan:
Jangan upload folder ZIP langsung ke GitHub. Upload isi foldernya.
