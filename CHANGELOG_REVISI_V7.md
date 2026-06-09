# Changelog Revisi V7

## Perbaikan OCR Nilai Gizi

- Menambahkan fungsi `repair_gram_unit_read_as_nine()` di `ocr_utils.py`.
- Menambahkan guard untuk kasus huruf `g` yang terbaca sebagai angka `9`.
- Memperbaiki contoh kasus:
  - `59` menjadi `5 g`
  - `259` menjadi `2.5 g`
  - `149` menjadi `14 g`
  - `19` menjadi `1 g`
  - `09` menjadi `0 g`
- Koreksi hanya berlaku untuk field gram seperti lemak total, lemak jenuh, protein, karbohidrat, gula, dan garam.
- Natrium mg dan energi kkal tidak ikut dikoreksi agar tidak merusak angka valid.

## Perbaikan Streamlit Session State

- Mengubah `input_form()` agar nilai default dimasukkan ke `st.session_state` sebelum widget dibuat.
- Menghapus pola pemberian `value=` bersamaan dengan `key=` pada widget OCR.
- Menambahkan pembersihan key lama seperti `ocr_saji`, `ocr_energi`, dan key legacy lain.
- Tujuannya menghilangkan warning: widget dibuat dengan default value tetapi juga diset lewat Session State API.

## Integritas Analisis

- OCR tetap hanya membantu mengisi data awal.
- Pengguna tetap harus memeriksa data sebelum menekan tombol analisis.
- Klasifikasi risiko tetap memakai logika konsisten dari revisi sebelumnya.
