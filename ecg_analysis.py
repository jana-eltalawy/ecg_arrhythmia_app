"""
============================================================================
ECG Arrhythmia Detection — Main Analysis Script
============================================================================
Applies custom BPF (1–4 kHz, Butterworth, MZT vs II) to ECG signals,
extracts features, trains classifier, and generates comparison plots.

Usage:
    python ecg_analysis.py
============================================================================
"""

import numpy as np
import scipy.signal as sig
from scipy.fft import rfft, rfftfreq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── CONFIGURATION ─────────────────────────────────────────
FS = 20050
F_LO, F_HI = 1000, 4000
N_LP = 4

# ═══════════════════════════════════════════════════════════
# FILTER DESIGN
# ═══════════════════════════════════════════════════════════

def design_filters(fs=FS, f_lo=F_LO, f_hi=F_HI, n_lp=N_LP):
    T = 1.0 / fs
    Omega_lo = 2 * np.pi * f_lo
    Omega_hi = 2 * np.pi * f_hi
    Omega_0 = np.sqrt(Omega_lo * Omega_hi)
    BW = Omega_hi - Omega_lo

    def eval_z(b, a, z):
        return sum(b[k] * z**(-k) for k in range(len(b))) / sum(a[k] * z**(-k) for k in range(len(a)))

    # MZT
    z_lp, p_lp, k_lp = sig.butter(n_lp, 1.0, btype='low', analog=True, output='zpk')
    z_bp, p_bp, k_bp = sig.lp2bp_zpk(z_lp, p_lp, k_lp, wo=Omega_0, bw=BW)
    dp = np.exp(p_bp * T)
    dz = np.concatenate([np.ones(n_lp), -np.ones(n_lp)])
    b_mzt = np.real(np.poly(dz)) * np.real(k_bp)
    a_mzt = np.real(np.poly(dp))
    fc = np.sqrt(f_lo * f_hi)
    b_mzt = b_mzt / abs(eval_z(b_mzt, a_mzt, np.exp(1j * 2 * np.pi * fc / fs)))

    # II
    z_lp2, p_lp2, k_lp2 = sig.butter(n_lp, 1.0, btype='low', analog=True, output='zpk')
    z_bp2, p_bp2, k_bp2 = sig.lp2bp_zpk(z_lp2, p_lp2, k_lp2, wo=Omega_0, bw=BW)
    b_bp, a_bp = sig.zpk2tf(z_bp2, p_bp2, k_bp2)
    r, p_pf, _ = sig.residue(b_bp, a_bp)

    pp = []
    used = [False] * len(p_pf)
    for i in range(len(p_pf)):
        if used[i]: continue
        for j in range(i + 1, len(p_pf)):
            if not used[j] and abs(p_pf[i] - np.conj(p_pf[j])) < 1:
                pp.append((r[i], p_pf[i], r[j], p_pf[j]))
                used[i] = used[j] = True
                break
        else:
            pp.append((r[i], p_pf[i], None, None))

    b_ii = np.array([1.0])
    a_ii = np.array([1.0])
    for (r1, p1, r2, p2) in pp:
        if r2 is None:
            zp = np.exp(p1 * T)
            b_sec = T * np.real([r1])
            a_sec = np.array([1.0, -zp])
        else:
            z1 = np.exp(p1 * T)
            z2 = np.exp(p2 * T)
            b0 = T * np.real(r1 + r2)
            b1 = -T * np.real(r1 * z2 + r2 * z1)
            a1 = -(z1 + z2)
            a2 = z1 * z2
            b_sec = np.real([b0, b1])
            a_sec = np.real([1.0, a1, a2])
        b_ii = np.polymul(b_ii, b_sec)
        a_ii = np.polymul(a_ii, a_sec)

    w_ii, H_ii = sig.freqz(b_ii, a_ii, worN=8192, fs=fs)
    b_ii = b_ii / (np.abs(H_ii[np.argmin(np.abs(w_ii - fc))]) + 1e-30)

    return (b_mzt, a_mzt), (b_ii, a_ii)

# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════

def extract_features(ecg_signal, b_mzt, a_mzt, b_ii, a_ii, fs=FS):
    ecg_mzt = sig.lfilter(b_mzt, a_mzt, ecg_signal)
    ecg_ii = sig.lfilter(b_ii, a_ii, ecg_signal)

    features = {}
    for prefix, sig_i in [('raw', ecg_signal), ('mzt', ecg_mzt), ('ii', ecg_ii)]:
        features[f'{prefix}_rms'] = np.sqrt(np.mean(sig_i**2))
        features[f'{prefix}_kurt'] = np.mean(sig_i**4) / (np.mean(sig_i**2)**2 + 1e-30)
        features[f'{prefix}_skew'] = np.mean(sig_i**3) / (np.mean(sig_i**2)**1.5 + 1e-30)

        peaks, _ = sig.find_peaks(sig_i, height=np.std(sig_i) * 2, distance=int(fs * 0.3))
        features[f'{prefix}_n_peaks'] = len(peaks)
        if len(peaks) > 1:
            rr = np.diff(peaks) / fs
            features[f'{prefix}_rr_mean'] = np.mean(rr)
            features[f'{prefix}_rr_std'] = np.std(rr)
            features[f'{prefix}_rr_cv'] = features[f'{prefix}_rr_std'] / (features[f'{prefix}_rr_mean'] + 1e-30)
        else:
            features[f'{prefix}_rr_mean'] = 0
            features[f'{prefix}_rr_std'] = 0
            features[f'{prefix}_rr_cv'] = 0

    for prefix, sig_i in [('mzt', ecg_mzt), ('ii', ecg_ii)]:
        fft_vals = np.abs(rfft(sig_i))
        freqs = rfftfreq(len(sig_i), 1 / fs)
        band_power = np.sum(fft_vals[(freqs >= F_LO) & (freqs <= F_HI)]**2)
        total_power = np.sum(fft_vals**2) + 1e-30
        features[f'{prefix}_band_ratio'] = band_power / total_power
        psd = fft_vals**2
        psd_norm = psd / (np.sum(psd) + 1e-30)
        features[f'{prefix}_spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-30))

    return features

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  ECG Arrhythmia Detection — Analysis Pipeline")
    print("=" * 60)

    # Load data
    data = np.load('data/ecg_dataset.npz', allow_pickle=True)
    train_ecg = data['train_ecg']
    train_types = data['train_types']
    test_ecg = data['test_ecg']
    test_types = data['test_types']

    # Design filters
    (b_mzt, a_mzt), (b_ii, a_ii) = design_filters()
    print("Filters designed: MZT + Impulse Invariance")

    # Extract features
    print("Extracting features...")
    train_features = []
    for ecg, lbl in zip(train_ecg, train_types):
        f = extract_features(ecg, b_mzt, a_mzt, b_ii, a_ii)
        f['label'] = lbl
        train_features.append(f)

    test_features = []
    for ecg, lbl in zip(test_ecg, test_types):
        f = extract_features(ecg, b_mzt, a_mzt, b_ii, a_ii)
        f['label'] = lbl
        test_features.append(f)

    # Prepare matrices
    feature_names = [k for k in train_features[0].keys() if k != 'label']
    X_train = np.array([[f[n] for n in feature_names] for f in train_features])
    y_train = np.array([f['label'] for f in train_features])
    X_test = np.array([[f[n] for n in feature_names] for f in test_features])
    y_test = np.array([f['label'] for f in test_features])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train
    print("Training classifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    # Save
    joblib.dump(clf, 'models/ecg_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    np.save('models/feature_names.npy', np.array(feature_names))
    print("\nModels saved to models/")
