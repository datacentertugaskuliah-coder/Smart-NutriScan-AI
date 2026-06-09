REVISI V10 SMART NUTRISCAN AI

Fokus revisi:
1. Menghapus hasil analisis dobel pada fitur Scan from Image.
2. Hasil analisis Scan from Image sekarang hanya tampil di panel kanan.
3. Fitur Analisis Produk Tunggal dibuat konsisten dengan layout berdampingan.
4. Panel kiri berisi form input atau konfirmasi OCR.
5. Panel kanan berisi Hasil Analisis AI atau Prediksi Risiko.
6. Hasil analisis tetap disimpan di session_state agar tidak hilang saat Streamlit rerun.

Cara upload:
1. Ekstrak ZIP.
2. Buka folder smart_nutriscan_revisi_v10.
3. Upload semua isi folder ke root repository GitHub.
4. Replace file lama.
5. Commit ke branch main.
6. Di Streamlit Cloud klik Clear cache.
7. Klik Reboot app.

Catatan logis:
Streamlit selalu melakukan rerun setelah tombol atau input berubah. Karena itu hasil analisis tidak boleh dirender langsung di bawah tombol sekaligus dirender di panel hasil. Revisi ini memakai satu sumber tampilan hasil, yaitu panel kanan.
