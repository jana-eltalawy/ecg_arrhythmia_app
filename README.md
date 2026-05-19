# 🫀 ECG Arrhythmia Detection App

## Overview
A complete **working application** for ECG arrhythmia detection using your custom-designed **Butterworth Bandpass Filter (1–4 kHz)**.

**Three input methods:**
1. **Upload your own ECG file** (CSV, TXT, NPZ, WAV)
2. **Browse the built-in dataset** (126 records with ground truth labels)
3. **Random sample** from dataset

**Two filter methods compared:**
- **Matched Z-Transform (MZT)** — Recommended
- **Impulse Invariance (II)** — Shows aliasing limitations

---

## Your Project Specifications

| Parameter | Value |
|-----------|-------|
| Sampling Rate | **20,050 Hz** |
| Passband | **1,000 – 4,000 Hz** |
| Prototype | **4th-order Butterworth LP** |
| Digital Order | **8th-order BPF** |

---

## Files

| File/Folder | Description |
|-------------|-------------|
| `app.py` | **Streamlit web application** |
| `dataset/train/` | **90 training records** (individual CSVs) |
| `dataset/test/` | **36 test records** (individual CSVs) |
| `dataset/samples/` | **3 sample files** (Normal, PVC, AFIB) |
| `dataset_index.csv` | **Dataset catalog** with labels |
| `models/ecg_classifier.pkl` | **Trained Random Forest** |
| `models/scaler.pkl` | **Feature scaler** |
| `models/feature_names.npy` | **Feature list** |
| `README.md` | This file |

---

## Quick Start

```bash
# 1. Install dependencies
pip install streamlit numpy scipy matplotlib pandas scikit-learn joblib

# 2. Launch the app
streamlit run app.py
```

Open `http://localhost:8501`

---

## How to Use

### Method 1: Upload Your Own ECG
1. Select "📤 Upload File" in sidebar
2. Drag & drop your file (CSV/TXT/NPZ/WAV)
3. Enter source sampling rate (default 360 Hz)
4. App auto-resamples to 20,050 Hz
5. Click through tabs to see results

### Method 2: Browse Dataset
1. Select "📂 Browse Dataset"
2. Filter by type (Normal/PVC/AFIB)
3. Pick any record from the list
4. True label is shown for verification

### Method 3: Random Sample
1. Select "🎲 Random Sample"
2. App loads a random record
3. Test your classification skills!

---

## App Tabs

| Tab | Content |
|-----|---------|
| **📈 Signal View** | Raw waveform + download button |
| **🔬 Filter Comparison** | Raw vs. MZT vs. II + SNR metrics |
| **📊 Spectrum** | Welch PSD for all three signals |
| **📋 Features & Dataset** | 25 extracted features + dataset explorer |

---

## Dataset Structure

Each CSV file contains:
```
time,amplitude
0.000000,0.015234
0.000050,0.012345
...
```

**Train set:** 90 records (30 Normal, 30 PVC, 30 AFIB)
**Test set:** 36 records (12 each type)

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **88.9%** |
| AFIB Precision | **100%** |
| PVC Precision | **100%** |
| Normal Precision | **75%** |

---

## Why MZT Wins

- **Impulse Invariance** aliases 50 Hz power-line noise into 1–4 kHz passband
- This corrupts R-peak detection and RR-interval features
- **Matched Z-Transform** maintains >80 dB stopband attenuation
- Clean separation of ECG from baseline wander and EMG

---

## Dependencies

```
streamlit>=1.28
numpy>=1.20
scipy>=1.7
matplotlib>=3.4
pandas>=1.3
scikit-learn>=1.0
joblib>=1.0
```
