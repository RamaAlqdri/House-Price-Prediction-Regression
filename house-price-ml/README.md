# House Price ML (End-to-End Regression)

Project ini melatih model regresi untuk prediksi harga rumah dari file CSV, mengevaluasi performa model, menyimpan model terbaik, dan melakukan prediksi pada data baru.

## Struktur Folder

```text
house-price-ml/
├─ data/
│  ├─ raw/
│  │  └─ houses.csv
│  └─ processed/
├─ models/
│  └─ model.joblib
├─ reports/
│  ├─ metrics.json
│  └─ predictions.csv
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ data.py
│  ├─ train.py
│  ├─ predict.py
│  └─ utils.py
├─ requirements.txt
└─ README.md
```

## Prasyarat

- Python 3.9+ (disarankan 3.10+)
- `pip`

## Setup Environment

Jalankan dari root project `house-price-ml`:

```bash
cd house-price-ml
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Jika di mesin kamu perintah `python` tidak tersedia, gunakan `python3`.

## Dataset

- Default data training: `data/raw/houses.csv`
- Default target: `price`

Jika kolom target di dataset kamu bukan `price`, jalankan training dengan `--target <nama_kolom_target>`.

## Training

Contoh command:

```bash
python -m src.train --data data/raw/houses.csv --target price
```

Output training:
- Melatih 3 model kandidat:
  - `linear_regression` (baseline)
  - `random_forest`
  - `gradient_boosting`
- Membersihkan target ke numerik; baris dengan target kosong/tidak valid akan dibuang otomatis.
- Mengabaikan fitur yang 100% missing saat training.
- Evaluasi train & test dengan metrik:
  - MAE
  - RMSE
  - R2
- Memilih model terbaik berdasarkan **RMSE test terendah**.
- Menyimpan:
  - Model terbaik: `models/model.joblib`
  - Ringkasan metrik: `reports/metrics.json`

## Prediksi Data Baru

Contoh command:

```bash
python -m src.predict --model models/model.joblib --input data/raw/new_data.csv --output reports/predictions.csv
```

Output:
- File `reports/predictions.csv` berisi semua kolom input + kolom baru `prediction`.

## Validasi Kolom Saat Prediksi

Script prediksi memeriksa apakah kolom minimal yang dipakai saat training tersedia pada data baru.

Jika ada kolom yang kurang, script akan gagal dengan error jelas, misalnya:
- "Input CSV is missing required feature columns from training..."

## Konfigurasi

Lihat `src/config.py` untuk default path dan parameter utama:
- `TARGET_COLUMN`
- `RANDOM_STATE`
- `TEST_SIZE`
- path model/report

## Troubleshooting

1. Error target column tidak ada
- Pastikan nama target benar.
- Jalankan ulang dengan `--target <nama_kolom_target>`.

2. Error file tidak ditemukan
- Cek path pada argumen `--data`, `--model`, atau `--input`.
- Pastikan command dijalankan dari folder `house-price-ml`.

3. Error kolom input tidak cocok saat prediksi
- Samakan skema kolom data prediksi dengan data training (minimal semua fitur training harus ada).

4. Install dependency gagal
- Pastikan virtual environment aktif.
- Ulangi:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

5. `python: command not found`
- Gunakan `python3` pada semua command (`python3 -m src.train`, `python3 -m src.predict`).
