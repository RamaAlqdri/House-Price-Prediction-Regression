# House Price ML (End-to-End Regression)

Repositori ini berisi pipeline machine learning untuk prediksi harga rumah dari data tabular campuran (numerik + kategorikal + teks angka). Fokus project ini adalah reproducible workflow: mulai dari data loading, preprocessing, training beberapa model, evaluasi metrik, simpan model terbaik, sampai prediksi data baru.

## 1. Tujuan Project

Tujuan utama:
- Membuat model regresi untuk memprediksi `Price (in rupees)`.
- Menyediakan pipeline training dan inference yang konsisten.
- Menyediakan notebook analisis untuk statistik data, visualisasi, dan review performa.

Output utama:
- `models/model.joblib`: artifact model terbaik + metadata.
- `reports/metrics.json`: ringkasan metrik train/test semua model kandidat.
- `reports/predictions.csv` atau `reports/predictions_from_notebook.csv`: hasil prediksi.

## 2. Struktur Folder

```text
house-price-ml/
├─ data/
│  ├─ raw/
│  │  ├─ houses.csv
│  │  ├─ _sample_200.csv
│  │  └─ new_data.csv (opsional)
│  └─ processed/
├─ models/
│  └─ model.joblib
├─ reports/
│  ├─ metrics.json
│  ├─ predictions.csv
│  └─ predictions_from_notebook.csv
├─ notebooks/
│  └─ house_price_e2e_review.ipynb
├─ src/
│  ├─ config.py
│  ├─ data.py
│  ├─ train.py
│  ├─ predict.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

## 3. Setup Environment

Jalankan dari root project `house-price-ml`:

```bash
cd house-price-ml
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Jika command `python` tersedia, kamu juga bisa pakai `python` menggantikan `python3`.

## 4. Konfigurasi Dasar

Lihat `src/config.py`:
- `TARGET_COLUMN = "Price (in rupees)"`
- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.2`
- Path default data/model/metrics.

Artinya secara default:
- Data training: `data/raw/houses.csv`
- Target prediksi: `Price (in rupees)`
- Split data: 80% train, 20% test

## 5. Alur Machine Learning dari Awal sampai Akhir

### 5.1 Definisi Problem

Problem yang diselesaikan adalah **supervised learning - regression**:
- Input: fitur properti (lokasi, luas, status, furnishing, dll).
- Output: nilai target kontinu `Price (in rupees)`.

### 5.2 Data Loading dan Validasi Dasar

Di `src/data.py`:
- File CSV dibaca dengan `load_data()`.
- Jika file tidak ada atau dataset kosong, proses dihentikan dengan error jelas.

### 5.3 Cleaning Target

Sebelum training:
- Target dipaksa menjadi numerik (`_coerce_target_to_numeric`).
- Nilai target yang kosong/tidak valid dibuang.
- Jumlah baris yang dibuang dilaporkan ke log.

Kenapa ini penting:
- Model regresi hanya bisa dilatih pada target numerik valid.
- Membersihkan target di awal menghindari error dan metrik menyesatkan.

### 5.4 Split Train/Test (Stratified Quantile)

`split_data()` melakukan:
- Pemisahan train/test.
- Stratifikasi berdasarkan quantile target (jika memungkinkan) untuk menjaga distribusi target train/test lebih seimbang.

Kenapa ini penting:
- Pada data harga rumah, distribusi sering skewed dan punya outlier.
- Stratifikasi membantu evaluasi lebih stabil dibanding split acak biasa.

### 5.5 Feature Engineering

Masih di `src/data.py` (`HouseFeatureEngineer`):
- Parsing teks angka ke numerik:
  - `Amount(in rupees)` (mis. Lac/Cr -> nilai numerik)
  - `Carpet Area`, `Super Area` (konversi satuan luas ke sqft)
  - `Floor`, `Bathroom`, `Balcony`, `Car Parking` (ekstraksi angka)
- Drop kolom noisy/high-cardinality:
  - `Title`, `Description`, `Index`

Tujuan:
- Mengurangi sparsity dan overfitting dari teks bebas.
- Meningkatkan kualitas sinyal numerik yang lebih robust.

### 5.6 Preprocessing

`build_preprocessor()` membangun pipeline campuran:
- Fitur numerik:
  - imputer median
  - standard scaler
- Fitur kategorikal:
  - imputer most-frequent
  - one-hot encoding (`handle_unknown='ignore'`)

Pipeline ini dipasang di semua model kandidat agar preprocessing konsisten.

### 5.7 Model Kandidat

Di `src/train.py`, model yang dilatih:
- `dummy_median`: baseline sederhana (prediksi median target).
- `linear_regression`: baseline linear.
- `random_forest`: ensemble tree.
- `random_forest_log_target`: random forest dengan transformasi target `log1p`.
- `gradient_boosting`: boosting tree.

### 5.8 Evaluasi

Metrik di `src/utils.py`:
- **MAE**: rata-rata error absolut.
- **RMSE**: akar dari rata-rata kuadrat error (lebih sensitif outlier).
- **R2**: proporsi variasi target yang dijelaskan model.

Aturan baca cepat:
- MAE lebih kecil lebih baik.
- RMSE lebih kecil lebih baik.
- R2 lebih besar lebih baik (mendekati 1 ideal; <= 0 berarti lemah).

### 5.9 Seleksi Model dan Simpan Artifact

Model terbaik dipilih berdasarkan **test RMSE terendah**.

Disimpan ke:
- `models/model.joblib`: berisi pipeline terlatih + metadata (`feature_columns`, `target_column`, dll).
- `reports/metrics.json`: metrik detail semua kandidat.

### 5.10 Inference (Prediksi Data Baru)

Di `src/predict.py`:
- Load artifact model.
- Validasi kolom input agar sesuai kolom training.
- Jalankan `pipeline.predict()`.
- Simpan output CSV dengan kolom baru `prediction`.

## 6. Cara Menjalankan

### 6.1 Training via Script

```bash
cd house-price-ml
source .venv/bin/activate
python -m src.train --data data/raw/houses.csv --target "Price (in rupees)"
```

### 6.2 Prediksi via Script

```bash
cd house-price-ml
source .venv/bin/activate
python -m src.predict \
  --model models/model.joblib \
  --input data/raw/new_data.csv \
  --output reports/predictions.csv
```

### 6.3 Analisis via Notebook

```bash
cd house-price-ml
source .venv/bin/activate
jupyter notebook notebooks/house_price_e2e_review.ipynb
```

Notebook mencakup:
- statistik deskriptif
- visualisasi missing values
- distribusi target
- perbandingan metrik model
- residual analysis
- feature importance (jika model mendukung)
- review otomatis hasil

### 6.4 Jalankan Flask API (untuk Frontend)

Pastikan dependency terpasang sesuai `requirements.txt` (terutama `scikit-learn==1.6.1` agar artifact model saat ini bisa di-load).

```bash
cd house-price-ml
source .venv/bin/activate
pip install -r requirements.txt
python -m src.api --model models/model.joblib --host 0.0.0.0 --port 5000
```

Endpoint yang tersedia:
- `GET /health` -> cek status service dan model.
- `POST /predict` -> prediksi harga rumah.

Contoh request single record:

```json
{
  "Amount(in rupees)": "35 Lac",
  "location": "Kolkata",
  "Carpet Area": "900 sqft",
  "Status": "Ready to Move",
  "Floor": "2 out of 5",
  "Transaction": "Resale",
  "Furnishing": "Semi-Furnished",
  "facing": "East",
  "overlooking": "Main Road",
  "Society": "Some Society",
  "Bathroom": "2",
  "Balcony": "1",
  "Car Parking": "1",
  "Ownership": "Freehold",
  "Super Area": "1000 sqft",
  "Title": "sample",
  "Description": "sample",
  "Index": 1
}
```

Contoh request batch:

```json
{
  "instances": [
    { "...": "record_1" },
    { "...": "record_2" }
  ]
}
```

Contoh response:

```json
{
  "count": 1,
  "prediction": 3412345.67,
  "predictions": [3412345.67]
}
```

## 7. Penjelasan Hasil (Snapshot `reports/metrics.json` Saat Ini)

Ringkasan data:
- Total baris: **187,531**
- Baris dipakai training/testing: **169,866**
- Baris dibuang (target invalid): **17,665**
- Test size: **20%**

Ranking performa test (lebih kecil RMSE lebih baik):

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| random_forest | 2,383.50 | 44,349.58 | 0.0060 |
| random_forest_log_target | 2,328.47 | 44,387.97 | 0.0043 |
| dummy_median | 3,495.36 | 44,519.84 | -0.0016 |
| linear_regression | 1,020.20 | 44,722.75 | -0.0108 |
| gradient_boosting | 2,555.23 | 44,857.56 | -0.0169 |

Model terpilih saat snapshot ini:
- **Best model: `random_forest`**
- Alasan: punya test RMSE terendah pada file metrik saat ini.

Interpretasi:
- Nilai **R2 test sangat rendah** (mendekati 0), artinya model baru menjelaskan sedikit variasi data test.
- Gap train-test pada beberapa model masih menunjukkan indikasi generalisasi belum kuat.
- Problem ini kemungkinan dipengaruhi kombinasi outlier target, keragaman data tinggi, dan fitur kategorikal besar.

Catatan:
- Angka di atas adalah snapshot dari file `reports/metrics.json` saat README ini ditulis.
- Jika kamu menjalankan training ulang, angka dapat berubah.

## 8. Cara Membaca File Output

### 8.1 `reports/metrics.json`

Isi penting:
- `models.<nama_model>.train/test`: MAE, RMSE, R2.
- `best_model`: model terbaik berdasarkan metric seleksi.
- `n_rows_dropped_target_invalid`: indikator kualitas target mentah.

### 8.2 `models/model.joblib`

Menyimpan:
- pipeline lengkap (feature engineering + preprocessing + model)
- daftar fitur training
- nama target
- metadata waktu training

### 8.3 `reports/predictions*.csv`

Berisi:
- seluruh kolom input
- kolom tambahan `prediction`

Gunakan file ini untuk validasi manual, dashboard, atau integrasi downstream.

## 9. Troubleshooting

1. Error `Target column not found`
- Pastikan nama kolom target benar.
- Jalankan dengan `--target "Nama Kolom Target"`.

2. Error `Input CSV is missing required feature columns`
- Samakan skema kolom file input prediksi dengan skema fitur saat training.

3. Error package/module di notebook
- Pastikan kernel Jupyter memakai interpreter `.venv` project.
- Jalankan `%pip install -r ../requirements.txt` dari notebook jika perlu.

4. Error saat load `model.joblib` (`AttributeError` / `InconsistentVersionWarning`)
- Biasanya karena mismatch versi `scikit-learn` antara training vs inference.
- Gunakan versi yang dipin di `requirements.txt` (saat ini `scikit-learn==1.6.1`) atau retrain model di environment versi baru lalu simpan ulang artifact.

5. Hasil test jelek (R2 rendah)
- Cek kualitas dan konsistensi fitur numerik.
- Lakukan cross-validation.
- Coba tuning hyperparameter lebih lanjut.

## 10. Rekomendasi Peningkatan Berikutnya

Prioritas teknis yang disarankan:
- Tambahkan **cross-validation** (KFold/Stratified by quantile) untuk metrik lebih stabil.
- Tambahkan **robust handling outlier** pada target (winsorize/capping atau evaluasi pada log-space).
- Uji model boosting yang lebih kuat (mis. XGBoost/LightGBM/CatBoost jika dependency memungkinkan).
- Tambahkan eksperimen fitur domain-specific (usia bangunan, jarak fasilitas, dsb jika data tersedia).
- Simpan run metadata per eksperimen (versi data, parameter, timestamp) agar tracking lebih rapi.

---

Jika kamu mau, saya bisa lanjutkan tahap berikutnya: menambahkan script evaluasi cross-validation + ringkasan leaderboard otomatis per eksperimen ke folder `reports/`.
