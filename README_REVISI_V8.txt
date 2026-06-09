REVISI V7 SMART NUTRISCAN AI

Fokus revisi:
1. Memperbaiki kasus huruf g pada label nilai gizi yang terbaca sebagai angka 9.
2. Menghapus warning Streamlit tentang widget default value dan Session State.
3. Menjaga alur OCR tetap aman, tidak otomatis berjalan saat gambar baru diupload.
4. Menjaga klasifikasi Aman, Sedang, Tinggi tetap konsisten dari revisi sebelumnya.

Perubahan logis OCR:
- 5g yang terbaca 59 akan dikoreksi menjadi 5.00 g.
- 2.5g yang terbaca 259 atau 2.59 akan dikoreksi menjadi 2.50 g.
- 14g yang terbaca 149 akan dikoreksi menjadi 14.00 g.
- 1g yang terbaca 19 akan dikoreksi menjadi 1.00 g.
- 0g yang terbaca 09 atau 9 dengan 0 persen AKG akan dikoreksi menjadi 0.00 g.
- Koreksi ini hanya diterapkan pada field berbasis gram, sehingga energi kkal dan natrium mg tidak ikut berubah.

Cara upload:
1. Ekstrak ZIP.
2. Upload semua isi folder smart_nutriscan_revisi_v7 ke root repository GitHub.
3. Replace file lama.
4. Commit ke branch main.
5. Di Streamlit Cloud, Clear cache.
6. Reboot app.

Catatan:
Hasil OCR tetap wajib dicek manual sebelum analisis karena OCR dari foto kemasan tidak pernah seratus persen pasti.
