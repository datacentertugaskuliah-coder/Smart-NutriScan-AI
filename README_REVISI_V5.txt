REVISI V5 SMART NUTRISCAN AI

Fokus revisi:
1. Memperbaiki OCR nilai gizi yang sebelumnya membaca angka tidak logis seperti 59 g, 259 g, dan 149 g.
2. Memperbaiki OCR komposisi agar tidak menggandakan hasil dari beberapa variasi preprocessing.
3. Menambahkan pembacaan Takaran Saji dari label.
4. Memisahkan variasi OCR terbaik dari variasi lain. Hasil mentah dari semua variasi tidak lagi digabung langsung.
5. Menambahkan guard logis untuk membedakan nilai gram, miligram, kilokalori, dan persen AKG.
6. Mencegah teks peringatan pada label menjadi nama produk.
7. Menyinkronkan hasil OCR ke widget form konfirmasi agar nilai yang tampil sama dengan data yang tersimpan.

Cara upload:
1. Ekstrak ZIP.
2. Upload semua file di dalam folder smart_nutriscan_revisi_v5 ke root repository GitHub.
3. Replace file lama.
4. Commit ke branch main.
5. Di Streamlit Cloud, pastikan Python 3.10 atau 3.11.
6. Clear cache.
7. Reboot app.

Catatan penggunaan:
1. Untuk nilai gizi, foto paling baik adalah crop khusus tabel Informasi Nilai Gizi.
2. Untuk komposisi, foto paling baik adalah crop khusus area Komposisi.
3. OCR tetap perlu dikoreksi manual sebelum analisis karena label pangan sering kecil, mengilap, melengkung, atau memuat dua bahasa.
4. Jika angka tidak terbaca dengan yakin, sistem akan membiarkan nilai tetap kosong daripada memasukkan angka yang menyesatkan.
