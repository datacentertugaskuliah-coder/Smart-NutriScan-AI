# Changelog Revisi V5

## Analisis masalah

Hasil OCR pada label nilai gizi sebelumnya tidak akurat karena sistem menggabungkan hasil dari beberapa variasi gambar secara mentah. Pada label kecil, hasil variasi `original`, `gray`, `sharpened`, dan `binary` bisa menghasilkan teks yang mirip tetapi angka berbeda. Jika semuanya digabung, parser bisa mengambil angka persen AKG atau angka rusak sebagai nilai gizi utama.

Contoh masalah yang diperbaiki:

- Takaran saji tetap 100 walaupun label menunjukkan 20 g.
- Energi tidak terbaca padahal label menunjukkan 100 kkal.
- Lemak total terbaca 59 padahal seharusnya sekitar 5 g.
- Lemak jenuh terbaca 259 padahal seharusnya sekitar 2.5 g.
- Karbohidrat terbaca 149 padahal seharusnya sekitar 14 g.
- Komposisi berulang karena hasil dari beberapa variasi preprocessing digabung.

## Perubahan teknis

1. `ocr_utils.py`
   - Menambahkan pemilihan variasi OCR terbaik.
   - Parsing nilai gizi sekarang memakai kandidat per variasi, bukan gabungan mentah.
   - Menambahkan pembacaan `takaran_saji`.
   - Menambahkan guard logis untuk satuan `g`, `mg`, `kkal`, dan `%`.
   - Menambahkan perbaikan desimal untuk angka OCR yang wajar rusak, misalnya `259` menjadi `2.59` pada lemak jenuh.
   - Komposisi sekarang diambil dari satu variasi terbaik dan dihentikan sebelum bagian `Ingredients` atau `Ingredienta` agar tidak dobel.
   - Nama produk tidak lagi diambil dari teks OCR nilai gizi agar teks peringatan tidak masuk sebagai nama produk.

2. `app.py`
   - Menambahkan sinkronisasi hasil OCR ke widget form konfirmasi.
   - Menambahkan informasi variasi gambar terbaik pada panel debug OCR.
   - Menambahkan pesan bahwa hanya angka yang lolos guard logis yang dimasukkan ke form.
   - Takaran saji sekarang bisa menerima hasil OCR.

## Prinsip revisi

OCR tidak boleh memaksa angka yang tidak masuk akal ke dalam form. Jika data tidak terbaca jelas, sistem harus lebih aman dengan membiarkan field kosong dan meminta koreksi manual.
