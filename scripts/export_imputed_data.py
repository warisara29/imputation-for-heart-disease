"""
บันทึกข้อมูลที่ impute แล้วทั้ง 35 ชุด (7 ระดับ x 5 วิธี) เป็น CSV
สำหรับนำไปใช้ใน SPSS
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# โหลดข้อมูล
df_complete = pd.read_csv('data/original/heart.csv')
target_col = 'target'
feature_cols = [c for c in df_complete.columns if c != target_col]

print(f"ชุดข้อมูล: {df_complete.shape[0]} แถว, {df_complete.shape[1]} คอลัมน์")

missing_proportions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

# สร้างโฟลเดอร์
os.makedirs('data/imputed_data', exist_ok=True)

# ============================================================
# ฟังก์ชัน
# ============================================================

def inject_mcar(data, proportion, feature_cols, random_state=42):
    rng = np.random.RandomState(random_state)
    data_missing = data.copy()
    n_rows = len(data)
    n_feat = len(feature_cols)
    n_missing = int(np.round(n_rows * n_feat * proportion))
    all_indices = [(i, col) for i in range(n_rows) for col in feature_cols]
    missing_indices = rng.choice(len(all_indices), size=n_missing, replace=False)
    for idx in missing_indices:
        row, col = all_indices[idx]
        data_missing.at[row, col] = np.nan
    return data_missing

def impute_mode(data, feature_cols):
    data_imp = data.copy()
    for col in feature_cols:
        mode_val = data[col].mode()
        if len(mode_val) > 0:
            data_imp[col] = data[col].fillna(mode_val[0])
    return data_imp

def impute_knn(data, feature_cols, k=20):
    data_imp = data.copy()
    data_np = data[feature_cols].values.astype(float)
    for i in range(len(data_np)):
        missing_cols = np.where(np.isnan(data_np[i]))[0]
        if len(missing_cols) == 0:
            continue
        observed_cols = np.where(~np.isnan(data_np[i]))[0]
        if len(observed_cols) == 0:
            for mc in missing_cols:
                col_vals = data_np[:, mc]
                valid = col_vals[~np.isnan(col_vals)]
                if len(valid) > 0:
                    vals, counts = np.unique(valid, return_counts=True)
                    data_np[i, mc] = vals[np.argmax(counts)]
            continue
        distances = []
        for j in range(len(data_np)):
            if i == j:
                continue
            shared = observed_cols[~np.isnan(data_np[j, observed_cols])]
            if len(shared) == 0:
                distances.append((j, float('inf')))
                continue
            hamming = np.sum(data_np[i, shared] != data_np[j, shared]) / len(shared)
            distances.append((j, hamming))
        distances.sort(key=lambda x: x[1])
        neighbors = [d[0] for d in distances[:k]]
        for mc in missing_cols:
            neighbor_vals = [data_np[n, mc] for n in neighbors if not np.isnan(data_np[n, mc])]
            if len(neighbor_vals) > 0:
                vals, counts = np.unique(neighbor_vals, return_counts=True)
                data_np[i, mc] = vals[np.argmax(counts)]
            else:
                col_vals = data_np[:, mc]
                valid = col_vals[~np.isnan(col_vals)]
                if len(valid) > 0:
                    vals, counts = np.unique(valid, return_counts=True)
                    data_np[i, mc] = vals[np.argmax(counts)]
    data_imp[feature_cols] = data_np
    return data_imp

def impute_rf(data, feature_cols, max_iter=10):
    data_np = data[feature_cols].values.astype(float)
    missing_mask = data[feature_cols].isna().values
    for c in range(data_np.shape[1]):
        col_vals = data_np[:, c]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) > 0:
            vals, counts = np.unique(valid, return_counts=True)
            data_np[np.isnan(data_np[:, c]), c] = vals[np.argmax(counts)]
    for iteration in range(max_iter):
        old_data = data_np.copy()
        for c in range(data_np.shape[1]):
            if not missing_mask[:, c].any():
                continue
            obs_rows = ~missing_mask[:, c]
            miss_rows = missing_mask[:, c]
            other_cols = [i for i in range(data_np.shape[1]) if i != c]
            X_train = data_np[obs_rows][:, other_cols]
            y_train = data_np[obs_rows, c].astype(int)
            X_pred = data_np[miss_rows][:, other_cols]
            if len(X_pred) == 0:
                continue
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            data_np[miss_rows, c] = rf.predict(X_pred)
        if np.sum(data_np != old_data) == 0:
            break
    data_imp = data.copy()
    data_imp[feature_cols] = data_np
    return data_imp

def impute_shd(data, feature_cols):
    data_imp = data.copy()
    data_imp = data_imp.sort_values(by=feature_cols[0], na_position='last').reset_index(drop=True)
    for col in feature_cols:
        last_observed = None
        for i in range(len(data_imp)):
            if pd.isna(data_imp.at[i, col]):
                if last_observed is not None:
                    data_imp.at[i, col] = last_observed
            else:
                last_observed = data_imp.at[i, col]
        first_observed = data_imp[col].dropna().iloc[0] if data_imp[col].dropna().shape[0] > 0 else 0
        data_imp[col] = data_imp[col].fillna(first_observed)
    return data_imp

def impute_mice(data, feature_cols, m=5, max_iter=10):
    imputed_datasets = []
    for ds in range(m):
        data_np = data[feature_cols].values.astype(float)
        missing_mask = data[feature_cols].isna().values
        rng = np.random.RandomState(42 + ds)
        for c in range(data_np.shape[1]):
            valid = data_np[:, c][~np.isnan(data_np[:, c])]
            if len(valid) > 0:
                miss_idx = np.where(np.isnan(data_np[:, c]))[0]
                data_np[miss_idx, c] = rng.choice(valid, size=len(miss_idx))
        for iteration in range(max_iter):
            for c in range(data_np.shape[1]):
                if not missing_mask[:, c].any():
                    continue
                obs_rows = ~missing_mask[:, c]
                miss_rows = missing_mask[:, c]
                other_cols = [i for i in range(data_np.shape[1]) if i != c]
                X_train = data_np[obs_rows][:, other_cols]
                y_train = data_np[obs_rows, c].astype(int)
                X_pred = data_np[miss_rows][:, other_cols]
                if len(X_pred) == 0:
                    continue
                rf = RandomForestClassifier(n_estimators=100, random_state=42 + ds + iteration, n_jobs=-1)
                rf.fit(X_train, y_train)
                proba = rf.predict_proba(X_pred)
                classes = rf.classes_
                drawn = np.array([rng.choice(classes, p=p) for p in proba])
                data_np[miss_rows, c] = drawn
        imp_df = data.copy()
        imp_df[feature_cols] = data_np
        imputed_datasets.append(imp_df)
    return imputed_datasets

# ============================================================
# สร้างและบันทึกข้อมูล imputed ทั้ง 35 ชุด
# ============================================================

for prop in missing_proportions:
    pct = int(prop * 100)
    print(f"\n{'='*50}")
    print(f"Missing {pct}%")
    print(f"{'='*50}")

    df_missing = inject_mcar(df_complete, prop, feature_cols)

    # Mode
    print(f"  [1/5] Mode...", end=" ")
    imp = impute_mode(df_missing, feature_cols)
    fname = f"data/imputed_data/imputed_mode_{pct}pct.csv"
    imp.to_csv(fname, index=False)
    print(f"✓ → {fname}")

    # KNN
    print(f"  [2/5] KNN...", end=" ")
    imp = impute_knn(df_missing, feature_cols, k=20)
    fname = f"data/imputed_data/imputed_knn_{pct}pct.csv"
    imp.to_csv(fname, index=False)
    print(f"✓ → {fname}")

    # RF
    print(f"  [3/5] Random Forest...", end=" ")
    imp = impute_rf(df_missing, feature_cols)
    fname = f"data/imputed_data/imputed_rf_{pct}pct.csv"
    imp.to_csv(fname, index=False)
    print(f"✓ → {fname}")

    # SHD
    print(f"  [4/5] Sequential Hot-Deck...", end=" ")
    imp = impute_shd(df_missing, feature_cols)
    fname = f"data/imputed_data/imputed_shd_{pct}pct.csv"
    imp.to_csv(fname, index=False)
    print(f"✓ → {fname}")

    # MICE (ใช้ชุดแรกจาก m=5)
    print(f"  [5/5] MICE...", end=" ")
    mice_list = impute_mice(df_missing, feature_cols, m=5)
    fname = f"data/imputed_data/imputed_mice_{pct}pct.csv"
    mice_list[0].to_csv(fname, index=False)
    print(f"✓ → {fname}")

print(f"\n{'='*50}")
print(f"เสร็จ! บันทึกทั้งหมด 35 ไฟล์ใน data/imputed_data/")
print(f"{'='*50}")
print("\nวิธีใช้ใน SPSS:")
print("1. File → Open → Data → เลือกไฟล์ CSV")
print("2. Analyze → Regression → Binary Logistic")
print("3. Dependent: target")
print("4. Covariates: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal")
print("5. กด OK → อ่านผลลัพธ์")
