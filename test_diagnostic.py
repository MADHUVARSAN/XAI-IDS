import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)

TRAIN = r"C:\Users\HP\Downloads\UNSW_NB15_training-set.csv"
TEST = r"C:\Users\HP\Downloads\UNSW_NB15_testing-set.csv"

print("=" * 70)
print("XAI-IDS OFFICIAL TEST SET DIAGNOSTIC")
print("=" * 70)

print("\nLoading datasets...")

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

print("Training shape :", train.shape)
print("Testing shape  :", test.shape)

print("\n" + "-" * 70)
print("TARGET DISTRIBUTION")
print("-" * 70)

print("\nTRAIN:")
print(train["label"].value_counts())
print(train["label"].value_counts(normalize=True).mul(100).round(2))

print("\nTEST:")
print(test["label"].value_counts())
print(test["label"].value_counts(normalize=True).mul(100).round(2))

print("\n" + "-" * 70)
print("COLUMN COMPARISON")
print("-" * 70)

train_cols = set(train.columns)
test_cols = set(test.columns)

print("Training columns:", len(train_cols))
print("Testing columns :", len(test_cols))

print("\nOnly in TRAIN:")
print(sorted(train_cols - test_cols))

print("\nOnly in TEST:")
print(sorted(test_cols - train_cols))

print("\n" + "-" * 70)
print("DUPLICATE ANALYSIS")
print("-" * 70)

print(
    "Training duplicate rows:",
    train.duplicated().sum()
)

print(
    "Testing duplicate rows :",
    test.duplicated().sum()
)

# Compare duplicate feature rows after removing target/leakage columns
drop_cols = [
    c for c in ["label", "attack_cat", "id"]
    if c in train.columns
]

train_features = train.drop(
    columns=drop_cols,
    errors="ignore"
)

test_features = test.drop(
    columns=drop_cols,
    errors="ignore"
)

common_cols = [
    c for c in train_features.columns
    if c in test_features.columns
]

train_features = train_features[common_cols]
test_features = test_features[common_cols]

print(
    "Common feature columns:",
    len(common_cols)
)

# Convert to string representation for robust duplicate matching
train_hash = pd.util.hash_pandas_object(
    train_features.astype(str),
    index=False
)

test_hash = pd.util.hash_pandas_object(
    test_features.astype(str),
    index=False
)

train_hash_set = set(train_hash)

overlap = sum(
    h in train_hash_set
    for h in test_hash
)

print(
    "Exact feature-row overlap:",
    overlap,
    f"({overlap / len(test_hash) * 100:.2f}%)"
)

print("\n" + "-" * 70)
print("NUMERIC FEATURE DISTRIBUTION")
print("-" * 70)

numeric_train = train_features.select_dtypes(
    include=np.number
)

numeric_test = test_features.select_dtypes(
    include=np.number
)

common_numeric = [
    c for c in numeric_train.columns
    if c in numeric_test.columns
]

distribution_rows = []

for col in common_numeric:

    tr = pd.to_numeric(
        numeric_train[col],
        errors="coerce"
    )

    te = pd.to_numeric(
        numeric_test[col],
        errors="coerce"
    )

    tr_mean = tr.mean()
    te_mean = te.mean()

    tr_std = tr.std()
    te_std = te.std()

    if pd.notna(tr_mean) and pd.notna(te_mean):
        mean_shift = abs(
            te_mean - tr_mean
        ) / (abs(tr_mean) + 1e-9)
    else:
        mean_shift = 0

    if pd.notna(tr_std) and pd.notna(te_std):
        std_shift = abs(
            te_std - tr_std
        ) / (abs(tr_std) + 1e-9)
    else:
        std_shift = 0

    distribution_rows.append({
        "feature": col,
        "train_mean": tr_mean,
        "test_mean": te_mean,
        "mean_shift": mean_shift,
        "train_std": tr_std,
        "test_std": te_std,
        "std_shift": std_shift,
    })

dist = pd.DataFrame(distribution_rows)

print("\nLargest MEAN shifts:")

print(
    dist.sort_values(
        "mean_shift",
        ascending=False
    ).head(15).to_string(index=False)
)

print("\nLargest STD shifts:")

print(
    dist.sort_values(
        "std_shift",
        ascending=False
    ).head(15).to_string(index=False)
)

print("\n" + "-" * 70)
print("CATEGORICAL DISTRIBUTION")
print("-" * 70)

categorical_cols = train_features.select_dtypes(
    exclude=np.number
).columns.tolist()

for col in categorical_cols:

    if col not in test_features.columns:
        continue

    train_values = set(
        train_features[col]
        .astype(str)
        .unique()
    )

    test_values = set(
        test_features[col]
        .astype(str)
        .unique()
    )

    unseen = test_values - train_values

    print(
        f"{col}: "
        f"train_unique={len(train_values)}, "
        f"test_unique={len(test_values)}, "
        f"unseen_test_values={len(unseen)}"
    )

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print(
    "\nNext: use these results to determine whether "
    "the validation/test gap is caused by distribution shift, "
    "feature mismatch, or dataset structure."
)