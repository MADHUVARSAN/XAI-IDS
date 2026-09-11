import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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


TRAIN_PATH = r"C:\Users\HP\Downloads\UNSW_NB15_training-set.csv"


def add_features(df):
    df = df.copy()

    eps = 1e-6

    if "sbytes" in df.columns and "dbytes" in df.columns:
        df["bytes_ratio"] = (
            df["sbytes"] + 1
        ) / (
            df["dbytes"] + 1
        )

        df["total_bytes"] = (
            df["sbytes"] + df["dbytes"]
        )

    if "spkts" in df.columns and "dpkts" in df.columns:
        df["packets_ratio"] = (
            df["spkts"] + 1
        ) / (
            df["dpkts"] + 1
        )

        df["total_packets"] = (
            df["spkts"] + df["dpkts"]
        )

    if "dur" in df.columns:
        if "sbytes" in df.columns:
            df["src_bytes_per_sec"] = (
                df["sbytes"]
                / (df["dur"] + eps)
            )

        if "dbytes" in df.columns:
            df["dst_bytes_per_sec"] = (
                df["dbytes"]
                / (df["dur"] + eps)
            )

        if "spkts" in df.columns:
            df["src_packets_per_sec"] = (
                df["spkts"]
                / (df["dur"] + eps)
            )

        if "dpkts" in df.columns:
            df["dst_packets_per_sec"] = (
                df["dpkts"]
                / (df["dur"] + eps)
            )

    if (
        "sload" in df.columns
        and "dload" in df.columns
    ):
        df["load_ratio"] = (
            df["sload"] + 1
        ) / (
            df["dload"] + 1
        )

        df["total_load"] = (
            df["sload"] + df["dload"]
        )

    if (
        "sttl" in df.columns
        and "dttl" in df.columns
    ):
        df["ttl_difference"] = (
            df["sttl"] - df["dttl"]
        ).abs()

    if (
        "smean" in df.columns
        and "dmean" in df.columns
    ):
        df["mean_packet_ratio"] = (
            df["smean"] + 1
        ) / (
            df["dmean"] + 1
        )

        df["mean_packet_difference"] = (
            df["smean"] - df["dmean"]
        ).abs()

    if (
        "sloss" in df.columns
        and "dloss" in df.columns
    ):
        df["total_loss"] = (
            df["sloss"] + df["dloss"]
        )

        df["loss_ratio"] = (
            df["sloss"] + 1
        ) / (
            df["dloss"] + 1
        )

    if (
        "tcprtt" in df.columns
        and "synack" in df.columns
    ):
        df["tcp_handshake_ratio"] = (
            df["synack"] + eps
        ) / (
            df["tcprtt"] + eps
        )

    return df


def prepare_data(df):
    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    y = pd.to_numeric(
        df["label"],
        errors="coerce"
    ).astype(int)

    X = df.drop(
        columns=["label"]
    )

    X = X.drop(
        columns=[
            c
            for c in X.columns
            if c.lower() in {
                "attack_cat",
                "id"
            }
        ],
        errors="ignore"
    )

    X = add_features(X)

    return X, y


def build_preprocessor(X):
    numeric = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical = [
        c
        for c in X.columns
        if c not in numeric
    ]

    numeric_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    return ColumnTransformer(
        [
            (
                "num",
                numeric_pipeline,
                numeric
            ),
            (
                "cat",
                categorical_pipeline,
                categorical
            )
        ]
    )


def evaluate(model, X, y):
    probability = model.predict_proba(
        X
    )[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy":
            accuracy_score(
                y,
                prediction
            ),
        "precision":
            precision_score(
                y,
                prediction,
                zero_division=0
            ),
        "recall":
            recall_score(
                y,
                prediction,
                zero_division=0
            ),
        "f1":
            f1_score(
                y,
                prediction,
                zero_division=0
            ),
        "fpr":
            fp / max(
                1,
                fp + tn
            ),
        "roc_auc":
            roc_auc_score(
                y,
                probability
            ),
        "pr_auc":
            average_precision_score(
                y,
                probability
            )
    }


def main():

    print()
    print("=" * 70)
    print("XAI-IDS FEATURE ENGINEERING EXPERIMENT")
    print("=" * 70)
    print()

    print("Loading UNSW-NB15...")

    df = pd.read_csv(
        TRAIN_PATH
    )

    print(
        "Original dataset:",
        df.shape
    )

    X, y = prepare_data(df)

    print(
        "Features after leakage removal:",
        X.shape[1]
    )

    print(
        "Class distribution:",
        y.value_counts().to_dict()
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print()
    print(
        "Train:",
        X_train.shape
    )

    print(
        "Validation:",
        X_val.shape
    )

    print()
    print("Building preprocessing...")

    preprocessor = build_preprocessor(
        X_train
    )

    X_train_t = preprocessor.fit_transform(
        X_train
    )

    X_val_t = preprocessor.transform(
        X_val
    )

    print(
        "Transformed train:",
        X_train_t.shape
    )

    print(
        "Transformed validation:",
        X_val_t.shape
    )

    configs = [
        {
            "name": "C",
            "n_estimators": 900,
            "max_depth": 7,
            "learning_rate": 0.025,
            "min_child_weight": 1,
            "gamma": 0.05,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
        },
        {
            "name": "F",
            "n_estimators": 1400,
            "max_depth": 6,
            "learning_rate": 0.015,
            "min_child_weight": 1,
            "gamma": 0.0,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
        },
        {
            "name": "G",
            "n_estimators": 1200,
            "max_depth": 7,
            "learning_rate": 0.015,
            "min_child_weight": 1,
            "gamma": 0.0,
            "subsample": 0.95,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
        },
        {
            "name": "H",
            "n_estimators": 1200,
            "max_depth": 8,
            "learning_rate": 0.015,
            "min_child_weight": 2,
            "gamma": 0.05,
            "subsample": 0.90,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.02,
            "reg_lambda": 1.2,
        },
    ]

    results = []

    for config in configs:

        print()
        print("-" * 70)
        print(
            "Testing configuration:",
            config["name"]
        )
        print("-" * 70)

        params = {
            k: v
            for k, v in config.items()
            if k != "name"
        }

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=4
        )

        start = time.perf_counter()

        model.fit(
            X_train_t,
            y_train
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        metrics = evaluate(
            model,
            X_val_t,
            y_val
        )

        result = {
            "name": config["name"],
            **metrics,
            "seconds": elapsed
        }

        results.append(
            result
        )

        print(
            "Accuracy :",
            f"{metrics['accuracy'] * 100:.2f}%"
        )

        print(
            "Precision:",
            f"{metrics['precision'] * 100:.2f}%"
        )

        print(
            "Recall   :",
            f"{metrics['recall'] * 100:.2f}%"
        )

        print(
            "F1       :",
            f"{metrics['f1'] * 100:.2f}%"
        )

        print(
            "FPR      :",
            f"{metrics['fpr'] * 100:.2f}%"
        )

        print(
            "ROC-AUC  :",
            f"{metrics['roc_auc'] * 100:.2f}%"
        )

        print(
            "PR-AUC   :",
            f"{metrics['pr_auc'] * 100:.2f}%"
        )

        print(
            "Time     :",
            f"{elapsed:.2f}s"
        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        by=[
            "accuracy",
            "precision",
            "f1"
        ],
        ascending=False
    )

    print()
    print()
    print("=" * 70)
    print("FEATURE ENGINEERING RESULTS")
    print("=" * 70)
    print()

    print(
        result_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Best configuration:",
        result_df.iloc[0]["name"]
    )

    print(
        "Best validation accuracy:",
        f"{result_df.iloc[0]['accuracy'] * 100:.2f}%"
    )

    print(
        "Best validation precision:",
        f"{result_df.iloc[0]['precision'] * 100:.2f}%"
    )

    print()
    print(
        "Experiment complete."
    )


if __name__ == "__main__":
    main()