import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv('heart.csv')
print(f"Original dataset: {df.shape}")
print(df.head())

# Target column should NOT have missing values (as per paper)
target_col = 'target'
feature_cols = [c for c in df.columns if c != target_col]

missing_proportions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

for prop in missing_proportions:
    df_missing = df.copy()
    n_rows = len(df)
    n_feat = len(feature_cols)
    n_total_cells = n_rows * n_feat
    n_missing = int(np.round(n_total_cells * prop))

    # Randomly select cells to set as NaN (MCAR mechanism)
    all_cells = [(i, col) for i in range(n_rows) for col in feature_cols]
    chosen = np.random.choice(len(all_cells), size=n_missing, replace=False)

    for idx in chosen:
        row, col = all_cells[idx]
        df_missing.at[row, col] = np.nan

    pct = int(prop * 100)
    filename = f'heart_missing_{pct}pct.csv'
    df_missing.to_csv(filename, index=False)

    actual = df_missing[feature_cols].isna().sum().sum()
    print(f"Saved {filename} — {actual}/{n_total_cells} cells missing ({actual/n_total_cells*100:.1f}%)")

print("\nDone! Generated 7 missing datasets.")
