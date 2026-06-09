# Changelog Revisi V6

## Analisis masalah

Pada revisi sebelumnya aplikasi dapat jatuh ke halaman `Oh no` setelah gambar diupload. Secara logis, kondisi ini bukan lagi error tampilan biasa. Penyebab yang paling mungkin adalah runtime OCR, beban memori, atau konflik state widget Streamlit.

Ada tiga titik rawan:

1. EasyOCR dimuat saat aplikasi pertama dibuka, padahal model OCR cukup berat.
2. Setelah OCR berhasil, kode lama mencoba menulis langsung ke `st.session_state` untuk key widget input yang sudah pernah dibuat.
3. Debug OCR menampilkan variasi gambar preprocessing, sehingga gambar hasil olahan ikut dikirim ke halaman dan bisa membebani memori.

## Perubahan teknis

1. `app.py`
   - EasyOCR dibuat lazy loading.
   - OCR hanya dimuat saat tombol proses OCR ditekan.
   - Update hasil OCR hanya dilakukan ke `st.session_state.ocr_data`.
   - Form konfirmasi memakai `ocr_form_version` agar widget dibuat ulang setelah OCR berhasil.
   - Debug variasi preprocessing hanya menampilkan nama variasi, bukan gambar besar.

2. `ocr_utils.py`
   - Ukuran maksimum gambar diturunkan dari 1700 menjadi 1100 piksel.
   - Pembesaran gambar kecil diturunkan dari 1.6 menjadi 1.25.
   - Jumlah variasi preprocessing aktif diturunkan dari 4 menjadi 2.

## Prinsip revisi

Aplikasi harus tetap hidup meskipun OCR gagal. OCR hanya alat bantu pengisian data. Keputusan klasifikasi tetap dijalankan setelah user mengoreksi dan mengonfirmasi data.
