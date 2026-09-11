import time
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA = r"C:\Users\HP\Downloads\UNSW_NB15_training-set.csv"
RANDOM_STATE = 42


def add_engineered_features(X):
    X = X.copy()

    def safe_ratio(a, b):
        return X[a] / (X[b].abs() + 1e-9)

    pairs = [
        ("bytes_ratio", "sbytes", "dbytes"),
        ("packets_ratio", "spkts", "dpkts"),
        ("load_ratio", "sload", "dload"),
        ("mean_packet_ratio", "smean", "dmean"),
        ("loss_ratio", "sloss", "dloss"),
    ]

    for new_col, a, b in pairs:
        if a in X.columns and b in X.columns:
            X[new_col] = safe_ratio(a, b)

    if "sbytes" in X.columns and "dbytes" in X.columns:
        X["total_bytes"] = X["sbytes"] + X["dbytes"]

    if "spkts" in X.columns and "dpkts" in X.columns:
        X["total_packets"] = X["spkts"] + X["dpkts"]

    if "sload" in X.columns and "dload" in X.columns:
        X["total_load"] = X["sload"] + X["dload"]

    if "sloss" in X.columns and "dloss" in X.columns:
        X["total_loss"] = X["sloss"] + X["dloss"]

    if "sttl" in X.columns and "dttl" in X.columns:
        X["ttl_difference"] = X["sttl"] - X["dttl"]

    if "smean" in X.columns and "dmean" in X.columns:
        X["mean_packet_difference"] = X["smean"] - X["dmean"]

    if "dur" in X.columns:
        dur = X["dur"].abs() + 1e-9

        for src, new_col in [
            ("sbytes", "src_bytes_per_sec"),
            ("dbytes", "dst_bytes_per_sec"),
            ("spkts", "src_packets_per_sec"),
            ("dpkts", "dst_packets_per_sec"),
        ]:
            if src in X.columns:
                X[new_col] = X[src] / dur

    if "synack" in X.columns and "tcprtt" in X.columns:
        X["tcp_handshake_ratio"] = X["synack"] / (
            X["tcprtt"].abs() + 1e-9
        )

    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    target = "label"

    if target not in df.columns:
        raise ValueError("label column not found.")

    y = pd.to_numeric(df[target], errors="coerce")

    X = df.drop(columns=[target], errors="ignore")

    # Critical leakage removal
    X = X.drop(
        columns=[
            c for c in X.columns
            if c.lower() in {"attack_cat", "id"}
        ],
        errors="ignore",
    )

    X = add_engineered_features(X)

    return X, y.astype(int)


def build_preprocessor(X):
    numeric_cols = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical_cols = [
        c for c in X.columns
        if c not in numeric_cols
    ]

    numeric_pipe = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ]

    categorical_pipe = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", __import__("sklearn").pipeline.Pipeline(numeric_pipe),
             numeric_cols),
            ("cat", __import__("sklearn").pipeline.Pipeline(categorical_pipe),
             categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def evaluate_model(model, X_val, y_val, threshold=0.50):
    prob = model.predict_proba(X_val)[:, 1]
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, pred
    ).ravel()

    return {
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(
            y_val, pred, zero_division=0
        ),
        "recall": recall_score(
            y_val, pred, zero_division=0
        ),
        "f1": f1_score(
            y_val, pred, zero_division=0
        ),
        "fpr": fp / (fp + tn),
        "roc_auc": roc_auc_score(y_val, prob),
        "pr_auc": average_precision_score(y_val, prob),
    }


print("=" * 70)
print("XAI-IDS FEATURE SELECTION EXPERIMENT")
print("=" * 70)

print("\nLoading UNSW-NB15...")

df = pd.read_csv(DATA)

print("Original dataset:", df.shape)

X, y = prepare_data(df)

print("Features after leakage removal + engineering:", X.shape[1])
print("Class distribution:", y.value_counts().to_dict())

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("\nTrain:", X_train.shape)
print("Validation:", X_val.shape)

print("\nBuilding preprocessing...")

preprocessor = build_preprocessor(X_train)

X_train_t = preprocessor.fit_transform(X_train)
X_val_t = preprocessor.transform(X_val)

feature_names = preprocessor.get_feature_names_out()

print("Transformed train:", X_train_t.shape)
print("Transformed validation:", X_val_t.shape)

# ------------------------------------------------------------------
# First train the H configuration on ALL transformed features.
# Then use XGBoost gain importance to select top features.
# ------------------------------------------------------------------

print("\n" + "-" * 70)
print("STEP 1: Training baseline H model")
print("-" * 70)

model_h = XGBClassifier(
    n_estimators=1200,
    max_depth=8,
    learning_rate=0.015,
    min_child_weight=2,
    gamma=0.05,
    subsample=0.90,
    colsample_bytree=0.95,
    reg_alpha=0.02,
    reg_lambda=1.2,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4,
)

start = time.time()

model_h.fit(
    X_train_t,
    y_train,
    verbose=False,
)

baseline_time = time.time() - start

baseline = evaluate_model(
    model_h,
    X_val_t,
    y_val,
)

print(f"Accuracy : {baseline['accuracy'] * 100:.2f}%")
print(f"Precision: {baseline['precision'] * 100:.2f}%")
print(f"Recall   : {baseline['recall'] * 100:.2f}%")
print(f"F1       : {baseline['f1'] * 100:.2f}%")
print(f"FPR      : {baseline['fpr'] * 100:.2f}%")
print(f"ROC-AUC  : {baseline['roc_auc'] * 100:.2f}%")
print(f"PR-AUC   : {baseline['pr_auc'] * 100:.2f}%")
print(f"Time     : {baseline_time:.2f}s")

# ------------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------------

importance = model_h.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False,
)

print("\n" + "-" * 70)
print("TOP 30 IMPORTANT FEATURES")
print("-" * 70)

print(
    importance_df.head(30).to_string(
        index=False
    )
)

# ------------------------------------------------------------------
# Feature selection experiments
# ------------------------------------------------------------------

feature_counts = [
    20,
    30,
    40,
    50,
    75,
    100,
    150,
]

results = []

for n_features in feature_counts:

    print("\n" + "-" * 70)
    print(f"Testing TOP {n_features} FEATURES")
    print("-" * 70)

    selected = importance_df.head(
        min(n_features, len(importance_df))
    )["feature"].tolist()

    selected_idx = [
        i for i, name in enumerate(feature_names)
        if name in selected
    ]

    Xtr = X_train_t[:, selected_idx]
    Xva = X_val_t[:, selected_idx]

    model = XGBClassifier(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.015,
        min_child_weight=2,
        gamma=0.05,
        subsample=0.90,
        colsample_bytree=0.95,
        reg_alpha=0.02,
        reg_lambda=1.2,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )

    start = time.time()

    model.fit(
        Xtr,
        y_train,
        verbose=False,
    )

    seconds = time.time() - start

    metrics = evaluate_model(
        model,
        Xva,
        y_val,
    )

    row = {
        "features": n_features,
        **metrics,
        "seconds": seconds,
    }

    results.append(row)

    print(f"Features : {n_features}")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1       : {metrics['f1'] * 100:.2f}%")
    print(f"FPR      : {metrics['fpr'] * 100:.2f}%")
    print(f"ROC-AUC  : {metrics['roc_auc'] * 100:.2f}%")
    print(f"PR-AUC   : {metrics['pr_auc'] * 100:.2f}%")
    print(f"Time     : {seconds:.2f}s")


results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FEATURE SELECTION RESULTS")
print("=" * 70)

print(
    results_df.sort_values(
        ["accuracy", "precision", "f1"],
        ascending=False,
    ).to_string(index=False)
)

best = results_df.sort_values(
    ["accuracy", "precision", "f1"],
    ascending=False,
).iloc[0]

print("\n" + "=" * 70)
print("BEST FEATURE SET")
print("=" * 70)

print(
    f"Top features : {int(best['features'])}"
)

print(
    f"Accuracy     : {best['accuracy'] * 100:.2f}%"
)

print(
    f"Precision    : {best['precision'] * 100:.2f}%"
)

print(
    f"Recall       : {best['recall'] * 100:.2f}%"
)

print(
    f"F1           : {best['f1'] * 100:.2f}%"
)

print(
    f"FPR          : {best['fpr'] * 100:.2f}%"
)

print(
    f"ROC-AUC      : {best['roc_auc'] * 100:.2f}%"
)

print(
    f"PR-AUC       : {best['pr_auc'] * 100:.2f}%"
)

print("\nExperiment complete.")