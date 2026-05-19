import streamlit as st
import numpy as np
import scipy.signal as sig
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from io import BytesIO
import os

st.set_page_config(page_title="ECG Arrhythmia Detector", layout="wide", page_icon="🫀")

# Load custom CSS (light theme)
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ── Custom CSS ────────────────────────────────────────────
# Inline dark CSS removed to favor light theme in assets/style.css

# ═══════════════════════════════════════════════════════════
# FILTER DESIGN
# ═══════════════════════════════════════════════════════════
@st.cache_data
def design_filters():
    fs = 20050
    f_lo, f_hi = 1000, 4000
    T = 1.0 / fs
    Omega_lo = 2 * np.pi * f_lo
    Omega_hi = 2 * np.pi * f_hi
    Omega_0 = np.sqrt(Omega_lo * Omega_hi)
    BW = Omega_hi - Omega_lo
    n_lp = 4

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

    pp = []; used = [False] * len(p_pf)
    for i in range(len(p_pf)):
        if used[i]: continue
        for j in range(i + 1, len(p_pf)):
            if not used[j] and abs(p_pf[i] - np.conj(p_pf[j])) < 1:
                pp.append((r[i], p_pf[i], r[j], p_pf[j]))
                used[i] = used[j] = True
                break
        else:
            pp.append((r[i], p_pf[i], None, None))

    b_ii = np.array([1.0]); a_ii = np.array([1.0])
    for (r1, p1, r2, p2) in pp:
        if r2 is None:
            zp = np.exp(p1 * T)
            b_sec = T * np.real([r1])
            a_sec = np.array([1.0, -zp])
        else:
            z1 = np.exp(p1 * T); z2 = np.exp(p2 * T)
            b0 = T * np.real(r1 + r2)
            b1 = -T * np.real(r1 * z2 + r2 * z1)
            a1 = -(z1 + z2); a2 = z1 * z2
            b_sec = np.real([b0, b1])
            a_sec = np.real([1.0, a1, a2])
        b_ii = np.polymul(b_ii, b_sec)
        a_ii = np.polymul(a_ii, a_sec)

    w_ii, H_ii = sig.freqz(b_ii, a_ii, worN=8192, fs=fs)
    b_ii = b_ii / (np.abs(H_ii[np.argmin(np.abs(w_ii - fc))]) + 1e-30)

    return (b_mzt, a_mzt), (b_ii, a_ii)

# ── LOAD MODELS ───────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        clf = joblib.load('models/ecg_classifier.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feat_names = np.load('models/feature_names.npy', allow_pickle=True)
        return clf, scaler, feat_names
    except:
        return None, None, None

# ── LOAD DATASET INDEX ───────────────────────────────────
@st.cache_data
def load_dataset_index():
    if os.path.exists('dataset_index.csv'):
        return pd.read_csv('dataset_index.csv')
    return pd.DataFrame()

def save_new_files_and_update_index(uploaded_files, category, rhythm_type):
    """
    Save new CSV files to training or testing directory and update index.csv
    """
    import csv
    index_file = 'dataset_index.csv'
    
    # Read existing index to avoid duplicates
    existing_files = set()
    if os.path.exists(index_file):
        with open(index_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_files.add(row['file'])
                
    new_rows = []
    os.makedirs(f'dataset/{category}', exist_ok=True)
    
    for f in uploaded_files:
        filename = f.name
        # avoid overwriting if duplicate
        base, ext = os.path.splitext(filename)
        counter = 1
        unique_filename = filename
        while unique_filename in existing_files:
            unique_filename = f"{base}_{counter}{ext}"
            counter += 1
            
        dest_path = f"dataset/{category}/{unique_filename}"
        with open(dest_path, "wb") as out_f:
            out_f.write(f.getbuffer())
            
        new_rows.append({
            'file': unique_filename,
            'folder': category,
            'type': rhythm_type,
            'path': dest_path
        })
        existing_files.add(unique_filename)
        
    # Write to CSV
    file_exists = os.path.exists(index_file)
    with open(index_file, 'a', newline='') as csvfile:
        fieldnames = ['file', 'folder', 'type', 'path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in new_rows:
            writer.writerow(r)
            
    return len(new_rows)

def retrain_model_on_files(df_index, b_mzt, a_mzt, b_ii, a_ii, progress_bar=None, status_text=None):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib

    train_df = df_index[df_index['folder'] == 'train']
    test_df = df_index[df_index['folder'] == 'test']

    if len(train_df) == 0 or len(test_df) == 0:
        return None, "Training or Test splits are empty in dataset_index.csv"

    train_features = []
    total_files = len(train_df) + len(test_df)
    processed = 0

    if status_text:
        status_text.text("Extracting features from Training Set...")

    for _, row in train_df.iterrows():
        filepath = row['path']
        if os.path.exists(filepath):
            try:
                sig_data = load_dataset_signal(filepath)
                feats, _, _ = extract_features(sig_data, b_mzt, a_mzt, b_ii, a_ii)
                feats['label'] = row['type']
                train_features.append(feats)
            except Exception as e:
                pass
        processed += 1
        if progress_bar:
            progress_bar.progress(processed / total_files)

    if status_text:
        status_text.text("Extracting features from Test Set...")

    test_features = []
    for _, row in test_df.iterrows():
        filepath = row['path']
        if os.path.exists(filepath):
            try:
                sig_data = load_dataset_signal(filepath)
                feats, _, _ = extract_features(sig_data, b_mzt, a_mzt, b_ii, a_ii)
                feats['label'] = row['type']
                test_features.append(feats)
            except Exception as e:
                pass
        processed += 1
        if progress_bar:
            progress_bar.progress(processed / total_files)

    if len(train_features) == 0 or len(test_features) == 0:
        return None, "No features could be extracted from files"

    feature_names = [k for k in train_features[0].keys() if k != 'label']
    X_train = np.array([[f[n] for n in feature_names] for f in train_features])
    y_train = np.array([f['label'] for f in train_features])
    X_test = np.array([[f[n] for n in feature_names] for f in test_features])
    y_test = np.array([f['label'] for f in test_features])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if status_text:
        status_text.text("Training RandomForest Classifier...")

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Save to disk
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, 'models/ecg_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    np.save('models/feature_names.npy', np.array(feature_names))

    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results = {
        'accuracy': acc,
        'report': report,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

    return results, None

# ── SIGNAL LOADER ─────────────────────────────────────────
def load_uploaded_signal(uploaded_file, target_fs=20050):
    file_type = uploaded_file.name.lower()

    if file_type.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        if 'amplitude' in df.columns:
            signal = df['amplitude'].values.astype(np.float32)
        else:
            signal = df.iloc[:, 0].values.astype(np.float32)
        source_fs = st.number_input("Source Sampling Rate (Hz)", 100, 100000, 360, key="up_fs")

    elif file_type.endswith('.txt'):
        signal = np.loadtxt(uploaded_file).astype(np.float32)
        if signal.ndim > 1: signal = signal[:, 0]
        source_fs = st.number_input("Source Sampling Rate (Hz)", 100, 100000, 360, key="up_fs")

    elif file_type.endswith('.npz'):
        data = np.load(uploaded_file, allow_pickle=True)
        keys = list(data.keys())
        signal_key = st.selectbox("Select array", keys, key="up_key")
        signal = data[signal_key].astype(np.float32)
        if signal.ndim > 1: signal = signal[:, 0]
        source_fs = st.number_input("Source Sampling Rate (Hz)", 100, 100000, 360, key="up_fs")

    elif file_type.endswith('.wav'):
        source_fs, signal = wavfile.read(BytesIO(uploaded_file.read()))
        signal = signal.astype(np.float32)
        if signal.ndim > 1: signal = signal[:, 0]
        st.info(f"WAV detected: {source_fs} Hz")
    else:
        st.error("Unsupported format. Use .csv, .txt, .npz, or .wav")
        return None

    if source_fs != target_fs:
        num_samples = int(len(signal) * target_fs / source_fs)
        signal = sig.resample(signal, num_samples)
        st.info(f"Resampled: {source_fs} Hz → {target_fs} Hz")

    return signal

def parse_ecg_signal(uploaded_file, source_fs=360, target_fs=20050):
    """
    Parse uploaded ECG files without displaying UI widgets
    """
    file_type = uploaded_file.name.lower()
    
    if file_type.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        if 'amplitude' in df.columns:
            signal = df['amplitude'].values.astype(np.float32)
        else:
            signal = df.iloc[:, 0].values.astype(np.float32)
    elif file_type.endswith('.txt'):
        signal = np.loadtxt(uploaded_file).astype(np.float32)
        if signal.ndim > 1: signal = signal[:, 0]
    elif file_type.endswith('.npz'):
        data = np.load(uploaded_file, allow_pickle=True)
        # pick first array
        key = list(data.keys())[0]
        signal = data[key].astype(np.float32)
        if signal.ndim > 1: signal = signal[:, 0]
    elif file_type.endswith('.wav'):
        source_fs, signal = wavfile.read(BytesIO(uploaded_file.read()))
        signal = signal.astype(np.float32)
        if signal.ndim > 1: signal = signal[:, 0]
    else:
        return None

    if source_fs != target_fs:
        num_samples = int(len(signal) * target_fs / source_fs)
        signal = sig.resample(signal, num_samples)
    return signal

def load_dataset_signal(filepath):
    """Load a signal from the dataset folder"""
    df = pd.read_csv(filepath)
    return df['amplitude'].values.astype(np.float32)

# ── FEATURE EXTRACTION ────────────────────────────────────
def extract_features(ecg_signal, b_mzt, a_mzt, b_ii, a_ii, fs=20050):
    ecg_mzt = sig.lfilter(b_mzt, a_mzt, ecg_signal)
    ecg_ii = sig.lfilter(b_ii, a_ii, ecg_signal)

    features = {}
    for prefix, sig_i in [('raw', ecg_signal), ('mzt', ecg_mzt), ('ii', ecg_ii)]:
        features[f'{prefix}_rms'] = np.sqrt(np.mean(sig_i**2))
        features[f'{prefix}_kurt'] = np.mean(sig_i**4) / (np.mean(sig_i**2)**2 + 1e-30)
        features[f'{prefix}_skew'] = np.mean(sig_i**3) / (np.mean(sig_i**2)**1.5 + 1e-30)

        peaks, _ = sig.find_peaks(sig_i, height=np.std(sig_i)*2, distance=int(fs*0.3))
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
        freqs = rfftfreq(len(sig_i), 1/fs)
        band_power = np.sum(fft_vals[(freqs >= 1000) & (freqs <= 4000)]**2)
        total_power = np.sum(fft_vals**2) + 1e-30
        features[f'{prefix}_band_ratio'] = band_power / total_power
        psd = fft_vals**2
        psd_norm = psd / (np.sum(psd) + 1e-30)
        features[f'{prefix}_spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-30))

    return features, ecg_mzt, ecg_ii

# ── PLOTTING ──────────────────────────────────────────────
def plot_signal(t, signal, title, color='white'):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, signal, color=color, lw=0.8)
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.set_title(title, color='#00d4ff', fontsize=12, fontweight='bold')
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.tick_params(colors='white')
    ax.grid(True, color='#30363d', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_spectrum(signal, fs, title, color='white'):
    f_vals, Pxx = sig.welch(signal, fs=fs, nperseg=4096, noverlap=2048)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy(f_vals, Pxx, color=color, lw=1.5)
    ax.axvline(1000, color='#ffb700', lw=1, ls='--', alpha=0.7)
    ax.axvline(4000, color='#39ff14', lw=1, ls='--', alpha=0.7)
    ax.axvline(50, color='#ff4757', lw=1, ls=':', alpha=0.5)
    ax.set_xlim(0, 6000); ax.set_ylim(1e-8, 1)
    ax.set_xlabel('Frequency (Hz)', color='white')
    ax.set_ylabel('PSD', color='white')
    ax.set_title(title, color=color, fontsize=12, fontweight='bold')
    ax.set_facecolor('#0d1117')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.tick_params(colors='white')
    ax.grid(True, color='#30363d', alpha=0.3)
    ax.legend(facecolor='#161b22', labelcolor='white', fontsize=9)
    plt.tight_layout()
    return fig

# ═══════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🫀 ECG Arrhythmia Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your signal OR select from dataset → Filter → Classify</div>', unsafe_allow_html=True)
st.markdown("---")

# Load resources
(b_mzt, a_mzt), (b_ii, a_ii) = design_filters()
clf, scaler, feat_names = load_models()
df_index = load_dataset_index()

# ═══════════════════════════════════════════════════════════
# SELECT SECTION / ROUTER
# ═══════════════════════════════════════════════════════════
app_mode = st.sidebar.selectbox("Select App Section:", ["🔍 Diagnostic Center", "⚙️ Model Studio & Retraining"])

if app_mode == "⚙️ Model Studio & Retraining":
    st.markdown("## ⚙️ Model Studio & Dataset Manager")
    st.markdown("Manage your training & testing files, upload new data, and retrain the machine learning classifier.")
    st.markdown("---")

    # Sidebar parameters for training
    st.sidebar.header("⚙️ Training Hyperparameters")
    n_estimators = st.sidebar.slider("RF Estimators (n_estimators)", 10, 300, 100, 10)
    max_depth = st.sidebar.slider("RF Max Depth", 2, 30, 10, 1)

    # Dataset splits summary
    train_count = len(df_index[df_index['folder'] == 'train']) if len(df_index) > 0 else 0
    test_count = len(df_index[df_index['folder'] == 'test']) if len(df_index) > 0 else 0
    sample_count = len(df_index[df_index['folder'] == 'samples']) if len(df_index) > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #0066cc; margin:0;">📁 Training Set</h4>
            <h2 style="color: #0066cc; margin-top:0.5rem; font-weight:800; margin-bottom: 0.5rem;">{train_count} Files</h2>
            <p style="color: #666; font-size: 0.9rem; margin:0;">Used to fit RF classifier</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #28a745; margin:0;">🧪 Test Set</h4>
            <h2 style="color: #28a745; margin-top:0.5rem; font-weight:800; margin-bottom: 0.5rem;">{test_count} Files</h2>
            <p style="color: #666; font-size: 0.9rem; margin:0;">Used to evaluate accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="color: #6f42c1; margin:0;">💡 Built-in Samples</h4>
            <h2 style="color: #6f42c1; margin-top:0.5rem; font-weight:800; margin-bottom: 0.5rem;">{sample_count} Files</h2>
            <p style="color: #666; font-size: 0.9rem; margin:0;">Quick demonstration samples</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for different operations
    m_tabs = st.tabs(["⚡ Retrain Model", "📤 Upload New Files", "📂 Dataset Browser"])

    with m_tabs[0]:
        st.markdown("### ⚡ Retrain Classifier")
        st.markdown("Extract features from all local CSV files and fit a new Random Forest classifier.")
        
        # Local function for retraining that uses sidebar hyperparameters
        def run_custom_retraining():
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            import joblib

            train_df = df_index[df_index['folder'] == 'train']
            test_df = df_index[df_index['folder'] == 'test']

            if len(train_df) == 0 or len(test_df) == 0:
                st.error("Training or Test splits are empty in dataset_index.csv")
                return

            p_bar = st.progress(0.0)
            status_txt = st.empty()
            
            train_features = []
            total_files = len(train_df) + len(test_df)
            processed = 0

            status_txt.text("Extracting features from Training Set...")
            for _, row in train_df.iterrows():
                filepath = row['path']
                if os.path.exists(filepath):
                    try:
                        sig_data = load_dataset_signal(filepath)
                        feats, _, _ = extract_features(sig_data, b_mzt, a_mzt, b_ii, a_ii)
                        feats['label'] = row['type']
                        train_features.append(feats)
                    except Exception as e:
                        pass
                processed += 1
                p_bar.progress(processed / total_files)

            status_txt.text("Extracting features from Test Set...")
            test_features = []
            for _, row in test_df.iterrows():
                filepath = row['path']
                if os.path.exists(filepath):
                    try:
                        sig_data = load_dataset_signal(filepath)
                        feats, _, _ = extract_features(sig_data, b_mzt, a_mzt, b_ii, a_ii)
                        feats['label'] = row['type']
                        test_features.append(feats)
                    except Exception as e:
                        pass
                processed += 1
                p_bar.progress(processed / total_files)

            if len(train_features) == 0 or len(test_features) == 0:
                st.error("No features could be successfully extracted from files.")
                return

            feature_names = [k for k in train_features[0].keys() if k != 'label']
            X_train = np.array([[f[n] for n in feature_names] for f in train_features])
            y_train = np.array([f['label'] for f in train_features])
            X_test = np.array([[f[n] for n in feature_names] for f in test_features])
            y_test = np.array([f['label'] for f in test_features])

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            status_txt.text("Training RandomForest Classifier...")
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            clf.fit(X_train_scaled, y_train)

            # Save models
            os.makedirs('models', exist_ok=True)
            joblib.dump(clf, 'models/ecg_classifier.pkl')
            joblib.dump(scaler, 'models/scaler.pkl')
            np.save('models/feature_names.npy', np.array(feature_names))

            # Evaluate
            from sklearn.metrics import accuracy_score, classification_report
            y_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            status_txt.text("Model training completed successfully!")
            st.success("🎉 Model retraining successfully completed!")
            st.balloons()
            
            # Clear resource/data cache
            st.cache_resource.clear()
            st.cache_data.clear()

            # Display Results
            res1, res2 = st.columns(2)
            with res1:
                st.metric("New Test Accuracy", f"{acc:.1%}")
                st.metric("Total Training Samples", len(X_train))
                st.metric("Total Test Samples", len(X_test))
            with res2:
                st.markdown("#### Classification Report")
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format(precision=3), use_container_width=True)

        if st.button("🚀 Start Model Retraining Pipeline", use_container_width=True):
            run_custom_retraining()

    with m_tabs[1]:
        st.markdown("### 📤 Upload New Dataset Files")
        st.markdown("Add your own custom CSV signals to the training or testing sets to expand the model's dataset.")
        
        with st.form("upload_form", clear_on_submit=True):
            up_files = st.file_uploader("Choose ECG CSV files", type=["csv"], accept_multiple_files=True)
            up_split = st.selectbox("Assign to Split:", ["train", "test"])
            up_type = st.selectbox("Rhythm Diagnosis Label:", ["Normal", "PVC", "AFIB"])
            submit_btn = st.form_submit_button("💾 Save Files to Dataset Split")
            
            if submit_btn:
                if not up_files:
                    st.warning("Please upload one or more CSV files.")
                else:
                    with st.spinner("Saving signals and rebuilding dataset index..."):
                        num_saved = save_new_files_and_update_index(up_files, up_split, up_type)
                        st.cache_data.clear() # clear cached data
                    st.success(f"Successfully added {num_saved} files to the {up_split} split under '{up_type}' category!")
                    st.rerun()

    with m_tabs[2]:
        st.markdown("### 📂 Dataset File Browser")
        if len(df_index) > 0:
            st.dataframe(df_index[['file', 'folder', 'type', 'path']], use_container_width=True, height=400)
        else:
            st.warning("Dataset index is empty.")

    st.stop()

# ═══════════════════════════════════════════════════════════
# SIDEBAR: INPUT SELECTION
# ═══════════════════════════════════════════════════════════

st.sidebar.header("📁 Input Selection")

# Tutorial toggle
show_tutorial = st.sidebar.checkbox("Show Tutorial", value=False)
if show_tutorial:
    st.markdown("## 📚 How to Use This App\n\n- **Upload** your ECG file (CSV, TXT, NPZ, or WAV).\n- The app will preprocess, extract features, and classify the rhythm.\n- View the waveform, spectrum, and confidence scores.\n- Download a **PDF report** with the results.\n- You can also explore the built‑in sample dataset via the **Browse Dataset** tab.")
    st.stop()

input_mode = st.sidebar.radio(
    "Choose input source:",
    ["📤 Upload File", "📂 Browse Dataset", "🎲 Random Sample"],
    index=1
)

signal = None
source_name = ""
true_label = None

if input_mode == "📤 Upload File":
    uploaded_files = st.sidebar.file_uploader("Upload ECG files for classification", type=['csv', 'txt', 'npz', 'wav'], accept_multiple_files=True)
    if uploaded_files:
        if len(uploaded_files) == 1:
            uploaded_file = uploaded_files[0]
            signal = load_uploaded_signal(uploaded_file)
            source_name = uploaded_file.name
            true_label = None
        else:
            # We are in batch upload mode!
            st.markdown("### 📊 Batch Classification Results")
            st.markdown("Features extracted and rhythms predicted for all uploaded files simultaneously.")
            st.markdown("---")
            
            # Let the user set the batch sampling rate
            batch_fs = st.sidebar.number_input("Batch Sampling Rate (Hz)", 100, 100000, 360)
            
            if clf is None or scaler is None:
                st.error("Classifier models are not loaded. Please train the model first.")
            else:
                results = []
                progress_bar = st.progress(0.0)
                status_txt = st.empty()
                
                for idx, f in enumerate(uploaded_files):
                    status_txt.text(f"Classifying file {idx+1}/{len(uploaded_files)}: {f.name}...")
                    try:
                        sig_data = parse_ecg_signal(f, source_fs=batch_fs)
                        if sig_data is not None:
                            feats, _, _ = extract_features(sig_data, b_mzt, a_mzt, b_ii, a_ii)
                            feat_vec = np.array([[feats[n] for n in feat_names]])
                            feat_scaled = scaler.transform(feat_vec)
                            pred = clf.predict(feat_scaled)[0]
                            proba = clf.predict_proba(feat_scaled)[0]
                            max_prob = proba[np.argmax(proba)]
                            
                            results.append({
                                'File Name': f.name,
                                'Predicted Rhythm': pred,
                                'Confidence': f"{max_prob:.1%}",
                                'Normal Prob': f"{proba[list(clf.classes_).index('Normal')]:.1%}" if 'Normal' in clf.classes_ else '0%',
                                'PVC Prob': f"{proba[list(clf.classes_).index('PVC')]:.1%}" if 'PVC' in clf.classes_ else '0%',
                                'AFIB Prob': f"{proba[list(clf.classes_).index('AFIB')]:.1%}" if 'AFIB' in clf.classes_ else '0%',
                            })
                    except Exception as e:
                        results.append({
                            'File Name': f.name,
                            'Predicted Rhythm': 'Error',
                            'Confidence': '0%',
                            'Normal Prob': '0%',
                            'PVC Prob': '0%',
                            'AFIB Prob': '0%'
                        })
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                status_txt.empty()
                progress_bar.empty()
                
                # Display Summary cards
                res_df = pd.DataFrame(results)
                normal_c = sum(res_df['Predicted Rhythm'] == 'Normal')
                pvc_c = sum(res_df['Predicted Rhythm'] == 'PVC')
                afib_c = sum(res_df['Predicted Rhythm'] == 'AFIB')
                
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #16a34a; margin:0;">✅ Normal Sinus</h4>
                        <h2 style="color: #16a34a; margin-top:0.5rem; font-weight:800; margin-bottom: 0.5rem;">{normal_c} Files</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #ea580c; margin:0;">⚠️ PVC Detected</h4>
                        <h2 style="color: #ea580c; margin-top:0.5rem; font-weight:800; margin-bottom: 0.5rem;">{pvc_c} Files</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #dc2626; margin:0;">🚨 AFIB Detected</h4>
                        <h2 style="color: #dc2626; margin-top:0.5rem; font-weight:800; margin-bottom: 0.5rem;">{afib_c} Files</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Show scannable dataframe
                st.dataframe(res_df, use_container_width=True)
                
                # Download results as CSV
                csv_data = res_df.to_csv(index=False)
                st.download_button("⬇️ Download Classification Report (CSV)", csv_data, "ecg_batch_classifications.csv", "text/csv", use_container_width=True)
                
            st.stop()


elif input_mode == "📂 Browse Dataset":
    if len(df_index) > 0:
        # Filter by type
        st.sidebar.markdown("**Filter by type:**")
        show_normal = st.sidebar.checkbox("Normal", True)
        show_pvc = st.sidebar.checkbox("PVC", True)
        show_afib = st.sidebar.checkbox("AFIB", True)

        filtered = df_index[
            ((df_index['type'] == 'Normal') & show_normal) |
            ((df_index['type'] == 'PVC') & show_pvc) |
            ((df_index['type'] == 'AFIB') & show_afib)
        ]

        if len(filtered) > 0:
            st.sidebar.markdown(f"**{len(filtered)} records available**")
            selected = st.sidebar.selectbox(
                "Select record:",
                filtered['file'].tolist(),
                format_func=lambda x: f"{x} ({filtered[filtered['file']==x]['type'].values[0]})"
            )
            if selected:
                filepath = filtered[filtered['file'] == selected]['path'].values[0]
                signal = load_dataset_signal(filepath)
                source_name = selected
                true_label = selected.split('_')[-1].replace('.csv', '')
        else:
            st.sidebar.warning("No records match filters")
    else:
        st.sidebar.error("Dataset index not found")

else:  # Random Sample
    if len(df_index) > 0:
        random_idx = np.random.randint(0, len(df_index))
        selected = df_index.iloc[random_idx]
        signal = load_dataset_signal(selected['path'])
        source_name = selected['file']
        true_label = selected['type']
        st.sidebar.success(f"Loaded: {source_name}")

# ═══════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════

if signal is not None:
    # Basic info
    st.markdown("### 📊 Signal Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Source", source_name[:20])
    with c2:
        st.metric("Samples", len(signal))
    with c3:
        st.metric("Duration", f"{len(signal)/20050:.2f} s")
    with c4:
        if true_label:
            st.metric("True Label", true_label)
        else:
            st.metric("True Label", "Unknown")

    # Process
    features, ecg_mzt, ecg_ii = extract_features(signal, b_mzt, a_mzt, b_ii, a_ii)

    # Classification
    if clf is not None and scaler is not None:
        feat_vec = np.array([[features[n] for n in feat_names]])
        feat_scaled = scaler.transform(feat_vec)
        pred = clf.predict(feat_scaled)[0]
        proba = clf.predict_proba(feat_scaled)[0]

        st.markdown("---")
        # Pre-define result texts
        if pred == 'Normal':
            rclass, ricon, rdesc = "result-normal", "✅", "Normal Sinus Rhythm"
        elif pred == 'PVC':
            rclass, ricon, rdesc = "result-pvc", "⚠️", "Premature Ventricular Contraction"
        else:
            rclass, ricon, rdesc = "result-afib", "🚨", "Atrial Fibrillation"

        st.markdown("---")
        st.markdown("### 🎯 Classification Result")

        # Result display
        res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{ricon} Detected Rhythm</h3>
                <div class="{rclass}">{pred}</div>
                <p style="color: #475569; margin-top: 0.5rem; font-weight: 500;">{rdesc}</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence
        st.markdown("#### Confidence Scores")
        conf_cols = st.columns(len(clf.classes_))
        for i, (cls, prob) in enumerate(zip(clf.classes_, proba)):
            color = '#16a34a' if cls=='Normal' else ('#d97706' if cls=='PVC' else '#dc2626')
            with conf_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color: {color}; margin: 0; font-weight: 700;">{cls}</h4>
                    <h2 style="color: {color}; margin-top: 0.5rem; font-weight: 800;">{prob:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)

        # PDF report generation below the metrics
        st.markdown("<br>", unsafe_allow_html=True)
        pdf_col1, pdf_col2, pdf_col3 = st.columns([1.5, 1, 1.5])
        with pdf_col2:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", size=12)
                pdf.cell(200, 10, txt="ECG Arrhythmia Detection Report", ln=1, align='C')
                pdf.ln(5)
                pdf.cell(200, 10, txt=f"Source file: {source_name}", ln=1)
                pdf.cell(200, 10, txt=f"Predicted rhythm: {pred} ({rdesc})", ln=1)
                pdf.cell(200, 10, txt=f"Confidence: {proba[np.argmax(proba)]:.1%}", ln=1)
                # Add simple feature table
                pdf.ln(5)
                pdf.cell(200, 10, txt="Extracted Features:", ln=1)
                for f_name, f_val in features.items():
                    pdf.cell(200, 8, txt=f"{f_name}: {f_val:.4f}", ln=1)
                pdf_bytes = pdf.output(dest='S').encode('latin1')
                st.download_button("⬇️ Download PDF Report", pdf_bytes, file_name=f"{source_name}_report.pdf", mime="application/pdf", use_container_width=True)

    st.markdown("---")

    # Visualization tabs
    tabs = st.tabs(["📈 Signal View", "🔬 Filter Comparison", "📊 Spectrum", "📋 Features & Dataset"])

    t = np.arange(len(signal)) / 20050

    with tabs[0]:
        fig = plot_signal(t, signal, f"Raw ECG — {source_name}", 'white')
        st.pyplot(fig, use_container_width=True)

        # Download raw signal
        csv = pd.DataFrame({'time': t, 'amplitude': signal}).to_csv(index=False)
        st.download_button("⬇️ Download Raw Signal (CSV)", csv, f"{source_name}_raw.csv", "text/csv")

    with tabs[1]:
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        axes[0].plot(t, signal, color='white', lw=0.8, alpha=0.6)
        axes[0].set_ylabel('Raw (mV)', color='white')
        axes[0].set_title('Raw ECG', color='#00d4ff', fontweight='bold')

        axes[1].plot(t, ecg_mzt, color='#bd93f9', lw=0.8)
        axes[1].set_ylabel('MZT (mV)', color='white')
        axes[1].set_title('Matched Z-Transform', color='#bd93f9', fontweight='bold')

        axes[2].plot(t, ecg_ii, color='#00d4ff', lw=0.8)
        axes[2].set_ylabel('II (mV)', color='white')
        axes[2].set_title('Impulse Invariance', color='#00d4ff', fontweight='bold')
        axes[2].set_xlabel('Time (s)', color='white')

        for ax in axes:
            ax.set_facecolor('#0d1117')
            for sp in ax.spines.values(): sp.set_color('#30363d')
            ax.tick_params(colors='white')
            ax.grid(True, color='#30363d', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # SNR metrics
        m1, m2, m3 = st.columns(3)
        snr_mzt = 10*np.log10(np.mean(ecg_mzt**2) / (np.var(signal-ecg_mzt) + 1e-30))
        snr_ii = 10*np.log10(np.mean(ecg_ii**2) / (np.var(signal-ecg_ii) + 1e-30))
        with m1: st.metric("MZT SNR", f"{snr_mzt:.1f} dB")
        with m2: st.metric("II SNR", f"{snr_ii:.1f} dB")
        with m3: st.metric("MZT Advantage", f"{snr_mzt-snr_ii:+.1f} dB")

    with tabs[2]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, sig_i, title, color in zip(axes, [signal, ecg_mzt, ecg_ii],
                                            ['Raw Spectrum', 'MZT Spectrum', 'II Spectrum'],
                                            ['white', '#bd93f9', '#00d4ff']):
            f_vals, Pxx = sig.welch(sig_i, fs=20050, nperseg=4096, noverlap=2048)
            ax.semilogy(f_vals, Pxx, color=color, lw=1.5)
            ax.axvline(1000, color='#ffb700', lw=1, ls='--', alpha=0.7)
            ax.axvline(4000, color='#39ff14', lw=1, ls='--', alpha=0.7)
            ax.axvline(50, color='#ff4757', lw=1, ls=':', alpha=0.5)
            ax.set_xlim(0, 6000); ax.set_ylim(1e-8, 1)
            ax.set_xlabel('Frequency (Hz)', color='white')
            ax.set_ylabel('PSD', color='white')
            ax.set_title(title, color=color, fontweight='bold')
            ax.set_facecolor('#0d1117')
            for sp in ax.spines.values(): sp.set_color('#30363d')
            ax.tick_params(colors='white')
            ax.grid(True, color='#30363d', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tabs[3]:
        # Features table
        st.markdown("#### Extracted Features")
        feat_df = pd.DataFrame({'Feature': list(features.keys()), 'Value': list(features.values())})
        st.dataframe(feat_df, use_container_width=True, height=400)

        # Dataset explorer
        st.markdown("---")
        st.markdown("#### 📂 Dataset Explorer")
        if len(df_index) > 0:
            st.dataframe(df_index[['file', 'folder', 'type']], use_container_width=True, height=300)

            # Download any dataset file
            dl_file = st.selectbox("Select file to download:", df_index['path'].tolist())
            if dl_file and os.path.exists(dl_file):
                with open(dl_file, 'rb') as f:
                    st.download_button("⬇️ Download Selected File", f, os.path.basename(dl_file), "text/csv")
        else:
            st.warning("Dataset index not found")

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; border: 2px dashed #30363d; border-radius: 15px;">
        <h2>👆 Select an Input Source</h2>
        <p>Upload your own ECG file, browse the dataset, or load a random sample</p>
        <p style="color: #666;">Supported: CSV, TXT, NPZ, WAV</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("ECG Arrhythmia Detection | Butterworth BPF 1–4 kHz | fs = 20,050 Hz | MZT vs. Impulse Invariance")
