# Changelog Revisi V9

## Perubahan utama

- Menambahkan penyimpanan hasil analisis OCR di `st.session_state.ocr_analysis_result`.
- Menambahkan penyimpanan hasil analisis manual di `st.session_state.manual_analysis_result`.
- Memecah proses analisis menjadi `build_analysis_result`, `render_analysis_result`, dan `run_product_analysis`.
- Menambahkan `make_analysis_signature` untuk membandingkan data input saat ini dengan data yang dipakai pada analisis terakhir.
- Mengubah layout Scan from Image menjadi dua kolom:
  - Kolom kiri: Konfirmasi Data Input (Hasil OCR)
  - Kolom kanan: Hasil Analisis AI (Prediksi Risiko)
- Menghapus pola hasil analisis yang hanya tampil sesaat setelah tombol diklik.
- Menghapus hasil analisis lama ketika OCR baru diproses atau hasil OCR direset.
- Menambahkan peringatan jika data input berubah setelah analisis terakhir.

## Prinsip integritas

Hasil analisis tidak lagi hilang karena rerun normal Streamlit. Namun, jika data input diubah setelah analisis, aplikasi tidak menganggap hasil lama sebagai hasil final. Sistem memberi peringatan agar pengguna menekan tombol analisis ulang.
