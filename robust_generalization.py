import time
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TRAIN = r"C:\Users\HP\Downloads\UNSW_NB15_training-set.csv"
TEST = r"C:\Users\HP\Downloads\UNSW_NB15_testing-set.csv"

RANDOM_STATE = 42


def add_features(X):
    X = X.copy()

    def ratio(a, b):
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
            X[new_col] = ratio(a, b)

    if "sbytes" in X and "dbytes" in X:
        X["total_bytes"] = X["sbytes"] + X["dbytes"]

    if "spkts" in X and "dpkts" in X:
        X["total_packets"] = X["spkts"] + X["dpkts"]

    if "sload" in X and "dload" in X:
        X["total_load"] = X["sload"] + X["dload"]

    if "sloss" in X and "dloss" in X:
        X["total_loss"] = X["sloss"] + X["dloss"]

    if "sttl" in X and "dttl" in X:
        X["ttl_difference"] = X["sttl"] - X["dttl"]

    if "smean" in X and "dmean" in X:
        X["mean_packet_difference"] = X["smean"] - X["dmean"]

    if "dur" in X:
        dur = X["dur"].abs() + 1e-9

        for src, new_col in [
            ("sbytes", "src_bytes_per_sec"),
            ("dbytes", "dst_bytes_per_sec"),
            ("spkts", "src_packets_per_sec"),
            ("dpkts", "dst_packets_per_sec"),
        ]:
            if src in X:
                X[new_col] = X[src] / dur

    if "synack" in X and "tcprtt" in X:
        X["tcp_handshake_ratio"] = (
            X["synack"] / (X["tcprtt"].abs() + 1e-9)
        )

    return X.replace([np.inf, -np.inf], np.nan)


def prepare(df, remove_features=None):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    y = pd.to_numeric(df["label"], errors="coerce").astype(int)

    X = df.drop(
        columns=["label", "attack_cat", "id"],
        errors="ignore",
    )

    X = add_features(X)

    if remove_features:
        X = X.drop(
            columns=[
                c for c in remove_features
                if c in X.columns
            ],
            errors="ignore",
        )

    return X, y


def build_preprocessor(X):
    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    categorical = [
        c for c in X.columns
        if c not in numeric
    ]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ])


def metrics(model, X, y, threshold=0.5):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred
    ).ravel()

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "recall": recall_score(
            y, pred, zero_division=0
        ),
        "f1": f1_score(
            y, pred, zero_division=0
        ),
        "fpr": fp / (fp + tn),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
    }


print("=" * 70)
print("XAI-IDS ROBUST GENERALIZATION EXPERIMENT")
print("=" * 70)

print("\nLoading UNSW-NB15...")

train_df = pd.read_csv(TRAIN)
test_df = pd.read_csv(TEST)

print("Training:", train_df.shape)
print("Testing :", test_df.shape)


experiments = {
    "ALL_FEATURES": [],
    "NO_STTL": ["sttl"],
    "NO_STTL_DTTL": ["sttl", "dttl"],
    "NO_HIGH_SHIFT": [
        "sttl",
        "dttl",
        "synack",
        "tcprtt",
        "ackdat",
        "dinpkt",
        "sjit",
        "dur",
        "response_body_len",
    ],
}


results = []


for name, removed in experiments.items():

    print("\n" + "=" * 70)
    print("EXPERIMENT:", name)
    print("=" * 70)

    print("Removed:", removed if removed else "None")

    X_train, y_train = prepare(
        train_df,
        removed,
    )

    X_test, y_test = prepare(
        test_df,
        removed,
    )

    # Same feature order
    common = [
        c for c in X_train.columns
        if c in X_test.columns
    ]

    X_train = X_train[common]
    X_test = X_test[common]

    # Internal validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X_tr)

    X_tr_t = preprocessor.fit_transform(X_tr)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    print(
        "Transformed:",
        X_tr_t.shape,
        X_val_t.shape,
        X_test_t.shape,
    )

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
        X_tr_t,
        y_tr,
        verbose=False,
    )

    train_time = time.time() - start

    val = metrics(
        model,
        X_val_t,
        y_val,
    )

    test = metrics(
        model,
        X_test_t,
        y_test,
    )

    print("\n--- INTERNAL VALIDATION ---")
    print(
        f"Accuracy : {val['accuracy'] * 100:.2f}%"
    )
    print(
        f"Precision: {val['precision'] * 100:.2f}%"
    )
    print(
        f"Recall   : {val['recall'] * 100:.2f}%"
    )
    print(
        f"F1       : {val['f1'] * 100:.2f}%"
    )
    print(
        f"FPR      : {val['fpr'] * 100:.2f}%"
    )

    print("\n--- OFFICIAL TEST ---")
    print(
        f"Accuracy : {test['accuracy'] * 100:.2f}%"
    )
    print(
        f"Precision: {test['precision'] * 100:.2f}%"
    )
    print(
        f"Recall   : {test['recall'] * 100:.2f}%"
    )
    print(
        f"F1       : {test['f1'] * 100:.2f}%"
    )
    print(
        f"FPR      : {test['fpr'] * 100:.2f}%"
    )
    print(
        f"ROC-AUC  : {test['roc_auc'] * 100:.2f}%"
    )
    print(
        f"PR-AUC   : {test['pr_auc'] * 100:.2f}%"
    )

    print(
        f"\nValidation → Test accuracy gap: "
        f"{(val['accuracy'] - test['accuracy']) * 100:.2f} percentage points"
    )

    results.append({
        "experiment": name,
        "removed": ",".join(removed) if removed else "None",
        "val_accuracy": val["accuracy"],
        "val_precision": val["precision"],
        "test_accuracy": test["accuracy"],
        "test_precision": test["precision"],
        "test_recall": test["recall"],
        "test_f1": test["f1"],
        "test_fpr": test["fpr"],
        "test_roc_auc": test["roc_auc"],
        "test_pr_auc": test["pr_auc"],
        "gap": val["accuracy"] - test["accuracy"],
        "seconds": train_time,
    })


print("\n" + "=" * 70)
print("ROBUST GENERALIZATION RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results)

display_cols = [
    "experiment",
    "val_accuracy",
    "val_precision",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_fpr",
    "test_roc_auc",
    "test_pr_auc",
    "gap",
]

print(
    results_df[
        display_cols
    ].to_string(index=False)
)

best = results_df.sort_values(
    [
        "test_accuracy",
        "test_precision",
        "test_f1",
    ],
    ascending=False,
).iloc[0]

print("\n" + "=" * 70)
print("BEST GENERALIZATION MODEL")
print("=" * 70)

print("Experiment:", best["experiment"])

print(
    f"Official Test Accuracy : "
    f"{best['test_accuracy'] * 100:.2f}%"
)

print(
    f"Official Test Precision: "
    f"{best['test_precision'] * 100:.2f}%"
)

print(
    f"Official Test Recall   : "
    f"{best['test_recall'] * 100:.2f}%"
)

print(
    f"Official Test F1       : "
    f"{best['test_f1'] * 100:.2f}%"
)

print(
    f"Official Test FPR      : "
    f"{best['test_fpr'] * 100:.2f}%"
)

print(
    f"Official ROC-AUC       : "
    f"{best['test_roc_auc'] * 100:.2f}%"
)

print(
    f"Official PR-AUC        : "
    f"{best['test_pr_auc'] * 100:.2f}%"
)

print("\nExperiment complete.")