# Proyek Analisis Data: Bike Sharing Dataset

**Nama:** Safira Salsabila  
**Email:** safirasalsabila.150306@gmail.com  
**ID Dicoding:** CDCC180D6X1053

---

## Deskripsi Proyek

Proyek ini melakukan analisis data komprehensif terhadap **Bike Sharing Dataset** dari Capital Bikeshare, Washington D.C. (2011–2012). Analisis mencakup dua granularitas data: harian (`day.csv`) dan per jam (`hour.csv`), meliputi proses data wrangling, exploratory data analysis (EDA), visualisasi explanatory, serta analisis lanjutan menggunakan clustering manual berbasis persentil.

---

## Struktur Direktori

```
submission/
├── dashboard/
│   ├── main_data.csv
│   └── dashboard.py
├── data/
│   ├── day.csv
│   └── hour.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── url.txt
```

---

## Pertanyaan Bisnis

1. Bagaimana pola rata-rata penyewaan berdasarkan musim dan kondisi cuaca selama 2011–2012?
2. Bagaimana perbandingan pengguna kasual vs terdaftar pada hari kerja vs non-kerja?
3. Pada jam berapa permintaan penyewaan mencapai puncaknya, dan apakah polanya berbeda antara hari kerja dan akhir pekan?

---

## Cara Menjalankan Dashboard

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Dashboard Streamlit

```bash
streamlit run dashboard/dashboard.py
```

### 3. Akses di Browser

Dashboard akan terbuka otomatis di: **http://localhost:8501**

---

## Dataset

- **Sumber:** [Bike Sharing Dataset — UCI ML Repository](http://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- **Periode:** 1 Januari 2011 – 31 Desember 2012
- **Granularitas:** Harian (day.csv, 731 baris) dan Per Jam (hour.csv, 17.544 baris)
