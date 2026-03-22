# A Comparison of Imputation Methods for Categorical Data

Replication of **Memon et al. (2023)** — *"A comparison of imputation methods for categorical data"* (Informatics in Medicine Unlocked) using the **Heart Disease Dataset** from Kaggle.

## Objective

Compare 5 imputation methods for handling missing data at varying levels of missingness (5%–50%) using the MCAR (Missing Completely at Random) mechanism.

## Imputation Methods

| Method | Description |
|---|---|
| **Mode** | Replace missing values with the most frequent value of each column |
| **KNN (K=20)** | Find 20 nearest neighbors using Hamming distance, impute with their mode |
| **Random Forest (RF)** | Iteratively predict missing values using Random Forest classifiers |
| **Sequential Hot-Deck (SHD)** | Sort data and fill missing values with the last observed value |
| **MICE (m=5)** | Multiple Imputation by Chained Equations using RF as univariate model |

## Evaluation

### Approach 1: Precision Score
- Proportion of imputed values that match the true (original) values
- Higher precision = better imputation accuracy

### Approach 2: AUC in 4 Classifiers
- Train classifiers on imputed data, evaluate prediction accuracy (AUC) on a held-out test set
- Classifiers: Logistic Regression, Random Forest, SVM, Naive Bayes

### Statistical Test
- **Kendall's W** test to assess consistency of imputation method rankings across missing data levels

## Dataset

- **Source:** [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Size:** 1,025 rows × 14 columns
- **Features:** 13 independent variables (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
- **Target:** heart disease (0 = no, 1 = yes)

## Project Structure

```
mini_project/
├── data/
│   ├── original/
│   │   └── heart.csv                        # Original complete dataset
│   └── missing_data/
│       ├── heart_missing_5pct.csv           # 5% MCAR missing data
│       ├── heart_missing_10pct.csv          # 10%
│       ├── heart_missing_15pct.csv          # 15%
│       ├── heart_missing_20pct.csv          # 20%
│       ├── heart_missing_30pct.csv          # 30%
│       ├── heart_missing_40pct.csv          # 40%
│       └── heart_missing_50pct.csv          # 50%
├── results/
│   ├── tables/
│   │   ├── results_precision_scores.csv     # Precision scores for all methods
│   │   ├── results_auc_logistic_regression.csv
│   │   ├── results_auc_random_forest.csv
│   │   ├── results_auc_svm.csv
│   │   └── results_auc_naive_bayes.csv
│   └── figures/
│       ├── fig2_precision_comparison.png    # Precision score comparison
│       ├── fig3_classifier_comparison.png   # AUC in 4 classifiers
│       └── fig4_average_auc.png             # Average AUC over 4 classifiers
├── scripts/
│   ├── generate_missing_data.py             # Generate MCAR missing datasets
│   ├── imputation_and_comparison.py         # Run imputation + evaluation
│   └── imputation_comparison_colab.ipynb    # Google Colab notebook (all-in-one)
├── imputation_for_categorical_data (2).pdf  # Reference paper
└── README.md
```

## How to Run

### Option 1: Google Colab (Recommended)
1. Upload `scripts/imputation_comparison_colab.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run all cells
3. Upload `heart.csv` when prompted
4. Results will be displayed and downloaded automatically

### Option 2: Local
```bash
cd mini_project

# Step 1: Generate missing datasets
python scripts/generate_missing_data.py

# Step 2: Run imputation and comparison
python scripts/imputation_and_comparison.py
```

**Requirements:** Python 3.8+, pandas, numpy, scikit-learn, scipy, matplotlib

## Key Results

### Precision Score (Approach 1)
- **RF imputation had the highest precision at all levels of missing data**
- SHD imputation had the lowest precision at all levels
- Kendall's W = 0.910 (strong consistency across missing data levels)

### Classifier AUC (Approach 2)
- **No single imputation method was universally best across all classifiers** — consistent with the paper's findings
- The best method varied depending on the classifier and the level of missing data

## Reference

Memon, S.MZ., Wamala, R., & Kabano, I.H. (2023). A comparison of imputation methods for categorical data. *Informatics in Medicine Unlocked*, 42, 101382. https://doi.org/10.1016/j.imu.2023.101382
