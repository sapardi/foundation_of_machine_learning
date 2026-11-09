"""
Script Eksperimen AI-Driven Data Loss Prevention (DLP) untuk BPJS Kesehatan
Mencakup:
1. Pembuatan Dataset Simulasi (jkn_claims_dataset.csv & network_access_logs.csv)
2. Klasifikasi Konten Sensitif Dokumen Klaim & Rekam Medis (TF-IDF + Random Forest)
3. Deteksi Anomali Jaringan & Eksfiltrasi Data (XGBoost)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import xgboost as xgb

def run_experiment():
    print("=== MEMULAI EKSPERIMEN AI-DRIVEN DLP UNTUK BPJS KESEHATAN ===")
    
    # 1. Pembuatan Dataset Simulasi Klaim & Rekam Medis
    np.random.seed(42)
    n_samples = 1000
    
    texts_sample = [
        "Data peserta JKN umum dengan nama Budi Santoso NIK 3201123456789001 diagnosis hipertensi ringan.",
        "Laporan keuangan operasional cabang BPJS Kesehatan periode Januari 2026.",
        "Rekam medis rahasia pasien HIV/AIDS dengan diagnosis ICD-10 B20 dan terapi ARV.",
        "Panduan layanan fasilitas kesehatan tingkat pertama (FKTP) tahun 2026.",
        "Database dump NIK peserta aktif seluruh Indonesia berjumlah 260 juta record (Confidential/Restricted).",
        "Surat edaran jam operasional kantor cabang BPJS Kesehatan."
    ]
    
    generated_texts = []
    generated_labels = []
    
    for i in range(n_samples):
        idx = i % len(texts_sample)
        text = texts_sample[idx] + f" [ID:{i}]"
        if "rahasia" in text or "dump" in text:
            label = "Restricted"
        elif "hipertensi" in text or "keuangan" in text:
            label = "Confidential"
        else:
            label = "Public"
        generated_texts.append(text)
        generated_labels.append(label)
        
    df_claims = pd.DataFrame({
        'document_text': generated_texts,
        'sensitivity_label': generated_labels
    })
    df_claims.to_csv('jkn_claims_dataset.csv', index=False)
    print("[1/3] Dataset Klaim & Rekam Medis berhasil dibuat: jkn_claims_dataset.csv (1000 baris)")

    # 2. Pembuatan Dataset Simulasi Log Jaringan & Eksfiltrasi
    n_logs = 2000
    query_freq = np.random.exponential(scale=10, size=n_logs)
    data_vol = np.random.exponential(scale=5, size=n_logs)
    hour = np.random.randint(0, 24, size=n_logs)
    failed_logins = np.random.poisson(lam=0.2, size=n_logs)
    
    is_exfiltration = np.where((query_freq > 40) | (data_vol > 25) | ((hour < 5) & (data_vol > 10)), 1, 0)
    
    df_network = pd.DataFrame({
        'query_frequency_per_min': query_freq,
        'data_volume_mb': data_vol,
        'hour_of_access': hour,
        'failed_login_attempts': failed_logins,
        'is_exfiltration': is_exfiltration
    })
    df_network.to_csv('network_access_logs.csv', index=False)
    print("[2/3] Dataset Network Logs berhasil dibuat: network_access_logs.csv (2000 baris)")

    # 3. Pelatihan dan Evaluasi Model Klasifikasi Konten (NLP / TF-IDF + Random Forest)
    X = df_claims['document_text']
    y = df_claims['sensitivity_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf_text = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_text.fit(X_train_vec, y_train)
    y_pred = clf_text.predict(X_test_vec)
    
    print("\n--- HASIL EVALUASI KLASIFIKASI KONTEN ---")
    print(classification_report(y_test, y_pred))
    print("Akurasi Model Klasifikasi:", accuracy_score(y_test, y_pred))

    # 4. Pelatihan dan Evaluasi Deteksi Anomali Jaringan (XGBoost)
    X_net = df_network[['query_frequency_per_min', 'data_volume_mb', 'hour_of_access', 'failed_login_attempts']]
    y_net = df_network['is_exfiltration']
    X_net_train, X_net_test, y_net_train, y_net_test = train_test_split(X_net, y_net, test_size=0.2, random_state=42, stratify=y_net)
    
    model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model_xgb.fit(X_net_train, y_net_train)
    
    y_net_pred = model_xgb.predict(X_net_test)
    y_net_proba = model_xgb.predict_proba(X_net_test)[:, 1]
    
    print("\n--- HASIL EVALUASI DETEKSI ANOMALI (XGBOOST) ---")
    print(classification_report(y_net_test, y_net_pred))
    print("ROC-AUC Score Deteksi Anomali:", roc_auc_score(y_net_test, y_net_proba))
    print("\n[3/3] Eksperimen selesai dengan sukses!")

if __name__ == "__main__":
    run_experiment()
