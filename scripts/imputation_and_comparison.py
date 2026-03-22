"""
ขั้นตอนที่ 3-4 ตาม Paper: เติมข้อมูลสูญหายด้วย 5 วิธี แล้วเปรียบเทียบผล
- Approach 1: เปรียบเทียบด้วย Precision Score
- Approach 2: เปรียบเทียบด้วย AUC ใน 4 Classifiers
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy

np.random.seed(42)

# ============================================================
# 1. โหลดข้อมูล
# ============================================================
df_complete = pd.read_csv('heart.csv')
target_col = 'target'
feature_cols = [c for c in df_complete.columns if c != target_col]

print(f"ชุดข้อมูลสมบูรณ์: {df_complete.shape[0]} แถว, {df_complete.shape[1]} คอลัมน์")

missing_proportions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
methods = ['Mode', 'KNN', 'RF', 'SHD', 'MICE']

# ============================================================
# 2. ฟังก์ชันสร้าง MCAR Missing Data
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

# ============================================================
# 3. ฟังก์ชัน Imputation 5 วิธี
# ============================================================

# วิธีที่ 1: Mode Imputation
def impute_mode(data, feature_cols):
    data_imp = data.copy()
    for col in feature_cols:
        mode_val = data[col].mode()
        if len(mode_val) > 0:
            data_imp[col] = data[col].fillna(mode_val[0])
    return data_imp

# วิธีที่ 2: KNN Imputation (K=20, Hamming distance)
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

# วิธีที่ 3: Random Forest Imputation (missForest algorithm)
def impute_rf(data, feature_cols, max_iter=10):
    data_np = data[feature_cols].values.astype(float)
    missing_mask = data[feature_cols].isna().values

    # เติมค่าเริ่มต้นด้วย mode
    for c in range(data_np.shape[1]):
        col_vals = data_np[:, c]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) > 0:
            vals, counts = np.unique(valid, return_counts=True)
            data_np[np.isnan(data_np[:, c]), c] = vals[np.argmax(counts)]

    # ทำซ้ำจนกว่าจะ converge
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

# วิธีที่ 4: Sequential Hot-Deck (SHD) Imputation
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

# วิธีที่ 5: MICE Imputation (m=5, ใช้ Random Forest เป็น univariate model)
def impute_mice(data, feature_cols, m=5, max_iter=10):
    imputed_datasets = []
    for ds in range(m):
        data_np = data[feature_cols].values.astype(float)
        missing_mask = data[feature_cols].isna().values
        rng = np.random.RandomState(42 + ds)

        # เติมค่าเริ่มต้นด้วยการสุ่มจากค่าที่มีอยู่
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
# 4. ฟังก์ชันคำนวณ Precision Score
# ============================================================
def calc_precision(original, imputed, missing_mask, feature_cols):
    correct = 0
    total = 0
    for col in feature_cols:
        mask = missing_mask[col]
        if mask.sum() == 0:
            continue
        correct += np.sum(original.loc[mask, col].values == imputed.loc[mask, col].values)
        total += mask.sum()
    return correct / total if total > 0 else 0

# ============================================================
# 5. Approach 1: เปรียบเทียบ Precision Score
# ============================================================
print("\n" + "=" * 70)
print("Approach 1: เปรียบเทียบวิธี Imputation ด้วย Precision Score")
print("=" * 70)

precision_results = {m: [] for m in methods}

for prop in missing_proportions:
    pct = int(prop * 100)
    print(f"\n--- Missing {pct}% ---")

    df_missing = inject_mcar(df_complete, prop, feature_cols)
    missing_mask = df_missing[feature_cols].isna()

    # Mode
    print("  [1/5] Mode...", end=" ")
    imp = impute_mode(df_missing, feature_cols)
    p = calc_precision(df_complete, imp, missing_mask, feature_cols)
    precision_results['Mode'].append(p)
    print(f"Precision = {p:.4f}")

    # KNN
    print("  [2/5] KNN (K=20)...", end=" ")
    imp = impute_knn(df_missing, feature_cols, k=20)
    p = calc_precision(df_complete, imp, missing_mask, feature_cols)
    precision_results['KNN'].append(p)
    print(f"Precision = {p:.4f}")

    # RF
    print("  [3/5] Random Forest...", end=" ")
    imp = impute_rf(df_missing, feature_cols)
    p = calc_precision(df_complete, imp, missing_mask, feature_cols)
    precision_results['RF'].append(p)
    print(f"Precision = {p:.4f}")

    # SHD
    print("  [4/5] Sequential Hot-Deck...", end=" ")
    imp = impute_shd(df_missing, feature_cols)
    p = calc_precision(df_complete, imp, missing_mask, feature_cols)
    precision_results['SHD'].append(p)
    print(f"Precision = {p:.4f}")

    # MICE
    print("  [5/5] MICE (m=5)...", end=" ")
    mice_dfs = impute_mice(df_missing, feature_cols, m=5)
    mice_precs = [calc_precision(df_complete, mdf, missing_mask, feature_cols) for mdf in mice_dfs]
    p = np.mean(mice_precs)
    precision_results['MICE'].append(p)
    print(f"Precision = {p:.4f}")

# ตาราง Precision
print("\n" + "=" * 70)
print("Table 2: Precision Score ของแต่ละวิธี")
print("=" * 70)
prec_df = pd.DataFrame(precision_results, index=[f"{int(p*100)}%" for p in missing_proportions])
prec_df.index.name = "Missing %"
print(prec_df.round(4).to_string())

# ตาราง Ranking
print("\n" + "=" * 70)
print("Table 3: อันดับของวิธี Imputation (1 = ดีที่สุด)")
print("=" * 70)
rank_df = prec_df.rank(axis=1, ascending=False).astype(int)
print(rank_df.to_string())
print(f"\nค่าเฉลี่ยอันดับ:\n{rank_df.mean().round(1)}")

# Kendall's W
ranks_matrix = rank_df.values
n_j, n_o = ranks_matrix.shape
rs = ranks_matrix.sum(axis=0)
mrs = np.mean(rs)
S = np.sum((rs - mrs) ** 2)
W = 12 * S / (n_j ** 2 * (n_o ** 3 - n_o))
chi_sq = n_j * (n_o - 1) * W
p_val = 1 - stats.chi2.cdf(chi_sq, n_o - 1)
print(f"\nKendall's W = {W:.3f}, Chi-sq = {chi_sq:.2f}, p = {p_val:.6f}")

# ============================================================
# 6. Approach 2: เปรียบเทียบ AUC ใน 4 Classifiers
# ============================================================
print("\n" + "=" * 70)
print("Approach 2: เปรียบเทียบด้วย AUC ใน 4 Classifiers")
print("=" * 70)

X = df_complete[feature_cols]
y = df_complete[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
print(f"Training: {len(train_df)}, Testing: {len(test_df)}")

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Naive Bayes': GaussianNB()
}

auc_results = {clf: {m: [] for m in methods} for clf in classifiers}

for prop in missing_proportions:
    pct = int(prop * 100)
    print(f"\n--- Missing {pct}% ---")

    train_missing = inject_mcar(train_df, prop, feature_cols)

    # Impute
    imputed = {}
    print("  Imputing...", end=" ")
    imputed['Mode'] = impute_mode(train_missing, feature_cols)
    print("Mode", end=" ")
    imputed['KNN'] = impute_knn(train_missing, feature_cols, k=20)
    print("KNN", end=" ")
    imputed['RF'] = impute_rf(train_missing, feature_cols)
    print("RF", end=" ")
    imputed['SHD'] = impute_shd(train_missing, feature_cols)
    print("SHD", end=" ")
    mice_list = impute_mice(train_missing, feature_cols, m=5)
    imputed['MICE'] = mice_list[0]
    print("MICE ✓")

    X_test_vals = test_df[feature_cols].values.astype(float)
    y_test_vals = test_df[target_col].values.astype(int)

    for clf_name, clf_template in classifiers.items():
        for method in methods:
            X_tr = imputed[method][feature_cols].values.astype(float)
            y_tr = imputed[method][target_col].values.astype(int)
            X_tr = np.nan_to_num(X_tr, nan=0)

            clf = deepcopy(clf_template)
            try:
                clf.fit(X_tr, y_tr)
                y_proba = clf.predict_proba(X_test_vals)[:, 1]
                auc = roc_auc_score(y_test_vals, y_proba)
            except:
                auc = 0.5
            auc_results[clf_name][method].append(auc)

    for clf_name in classifiers:
        aucs = [f"{m}={auc_results[clf_name][m][-1]:.4f}" for m in methods]
        print(f"  {clf_name}: {', '.join(aucs)}")

# ============================================================
# 7. ตาราง AUC และ Ranking สำหรับแต่ละ Classifier
# ============================================================
print("\n" + "=" * 70)
print("ผลลัพธ์: อันดับวิธี Imputation ตาม AUC ในแต่ละ Classifier")
print("=" * 70)

for clf_name in classifiers:
    print(f"\n{'─'*50}")
    print(f"  {clf_name}")
    print(f"{'─'*50}")

    auc_df = pd.DataFrame(auc_results[clf_name],
                          index=[f"{int(p*100)}%" for p in missing_proportions])
    print("AUC:")
    print(auc_df.round(4).to_string())

    rank_clf = auc_df.rank(axis=1, ascending=False).astype(int)
    print("\nอันดับ (1 = ดีที่สุด):")
    print(rank_clf.to_string())
    print(f"\nค่าเฉลี่ยอันดับ:\n{rank_clf.mean().round(1)}")

    rm = rank_clf.values
    nj, no = rm.shape
    rs = rm.sum(axis=0)
    S = np.sum((rs - np.mean(rs)) ** 2)
    W = 12 * S / (nj ** 2 * (no ** 3 - no))
    chi = nj * (no - 1) * W
    pv = 1 - stats.chi2.cdf(chi, no - 1)
    print(f"Kendall's W = {W:.3f}, Chi-sq = {chi:.2f}, p = {pv:.6f}")

# ============================================================
# 8. สร้างกราฟ
# ============================================================
x_vals = [int(p * 100) for p in missing_proportions]
colors = {'Mode': '#1f77b4', 'KNN': '#ff7f0e', 'RF': '#2ca02c', 'SHD': '#d62728', 'MICE': '#9467bd'}

# กราฟ 1: Precision Score (เทียบ Fig 2 ใน paper)
fig, ax = plt.subplots(figsize=(10, 6))
for method in methods:
    ax.plot(x_vals, precision_results[method], marker='o', label=method, linewidth=2, color=colors[method])
ax.set_xlabel('Proportion (%) of Missing Data', fontsize=12)
ax.set_ylabel('Precision Score', fontsize=12)
ax.set_title('Precision Score of Imputation Methods', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_precision_comparison.png', dpi=150)
print("\nบันทึก: fig2_precision_comparison.png")

# กราฟ 2: AUC ใน 4 Classifiers (เทียบ Fig 3 ใน paper)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, clf_name in enumerate(classifiers):
    ax = axes[idx]
    for method in methods:
        ax.plot(x_vals, auc_results[clf_name][method], marker='o', label=method, linewidth=2, color=colors[method])
    ax.set_xlabel('Proportion (%) of Missing Data', fontsize=10)
    ax.set_ylabel('AUC', fontsize=10)
    ax.set_title(clf_name, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle('Comparison of Imputation Methods by Prediction Accuracy', fontsize=14)
plt.tight_layout()
plt.savefig('fig3_classifier_comparison.png', dpi=150)
print("บันทึก: fig3_classifier_comparison.png")

# กราฟ 3: Average AUC (เทียบ Fig 4 ใน paper)
fig, ax = plt.subplots(figsize=(10, 6))
for method in methods:
    avg_aucs = [np.mean([auc_results[c][method][i] for c in classifiers]) for i in range(len(missing_proportions))]
    ax.plot(x_vals, avg_aucs, marker='o', label=method, linewidth=2, color=colors[method])
ax.set_xlabel('Proportion (%) of Missing Data', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('Average AUC over Four Classifiers', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_average_auc.png', dpi=150)
print("บันทึก: fig4_average_auc.png")

# บันทึกผลลัพธ์เป็น CSV
prec_df.to_csv('results_precision_scores.csv')
for clf_name in classifiers:
    auc_df = pd.DataFrame(auc_results[clf_name], index=[f"{int(p*100)}%" for p in missing_proportions])
    auc_df.to_csv(f'results_auc_{clf_name.replace(" ", "_").lower()}.csv')

print("\n" + "=" * 70)
print("เสร็จสมบูรณ์!")
print("=" * 70)
