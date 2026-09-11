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


def prepare_data(df):
    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    target = "label"

    y = pd.to_numeric(
        df[target],
        errors="coerce"
    ).astype(int)

    X = df.drop(
        columns=[target]
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


def evaluate_model(model, X_val, y_val):
    probability = model.predict_proba(
        X_val
    )[:, 1]

    prediction = (
        probability >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        prediction,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy":
            accuracy_score(
                y_val,
                prediction
            ),
        "precision":
            precision_score(
                y_val,
                prediction,
                zero_division=0
            ),
        "recall":
            recall_score(
                y_val,
                prediction,
                zero_division=0
            ),
        "f1":
            f1_score(
                y_val,
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
                y_val,
                probability
            ),
        "pr_auc":
            average_precision_score(
                y_val,
                probability
            )
    }


def main():
    print()
    print("=" * 70)
    print("XAI-IDS UNSW-NB15 VALIDATION TUNING")
    print("=" * 70)
    print()

    print("Loading training dataset...")

    df = pd.read_csv(
        TRAIN_PATH
    )

    print(
        "Dataset:",
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
        "Train split:",
        X_train.shape
    )

    print(
        "Validation split:",
        X_val.shape
    )

    print()
    print("Fitting preprocessing...")

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
        "Transformed train shape:",
        X_train_t.shape
    )

    print(
        "Transformed validation shape:",
        X_val_t.shape
    )

    positive = int(
        (y_train == 1).sum()
    )

    negative = int(
        (y_train == 0).sum()
    )

    scale_pos_weight = (
        negative / positive
    )

    print(
        "scale_pos_weight:",
        round(
            scale_pos_weight,
            4
        )
    )

    candidates = [
        {
            "name": "A",
            "n_estimators": 700,
            "max_depth": 5,
            "learning_rate": 0.035,
            "min_child_weight": 1,
            "gamma": 0.0,
            "subsample": 0.95,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.05,
            "reg_lambda": 1.2,
        },
        {
            "name": "B",
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.025,
            "min_child_weight": 1,
            "gamma": 0.0,
            "subsample": 0.95,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
        },
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
            "name": "D",
            "n_estimators": 1000,
            "max_depth": 8,
            "learning_rate": 0.02,
            "min_child_weight": 2,
            "gamma": 0.05,
            "subsample": 0.90,
            "colsample_bytree": 0.95,
            "reg_alpha": 0.02,
            "reg_lambda": 1.2,
        },
        {
            "name": "E",
            "n_estimators": 1200,
            "max_depth": 6,
            "learning_rate": 0.02,
            "min_child_weight": 2,
            "gamma": 0.1,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_alpha": 0.02,
            "reg_lambda": 1.5,
        }
    ]

    results = []

    for params in candidates:
        name = params["name"]

        print()
        print("-" * 70)
        print(
            "Testing configuration:",
            name
        )
        print("-" * 70)

        model_params = {
            k: v
            for k, v in params.items()
            if k != "name"
        }

        model = XGBClassifier(
            **model_params,
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

        metrics = evaluate_model(
            model,
            X_val_t,
            y_val
        )

        result = {
            "name": name,
            **metrics,
            "train_seconds": elapsed
        }

        results.append(result)

        print()
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
            "precision",
            "accuracy",
            "f1"
        ],
        ascending=False
    )

    print()
    print()
    print("=" * 70)
    print("FINAL VALIDATION COMPARISON")
    print("=" * 70)
    print()

    display_columns = [
        "name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "fpr",
        "roc_auc",
        "pr_auc",
        "train_seconds"
    ]

    print(
        result_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    best = result_df.iloc[0]

    print()
    print("=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)

    print(
        "Configuration:",
        best["name"]
    )

    print(
        "Validation Accuracy:",
        f"{best['accuracy'] * 100:.2f}%"
    )

    print(
        "Validation Precision:",
        f"{best['precision'] * 100:.2f}%"
    )

    print(
        "Validation Recall:",
        f"{best['recall'] * 100:.2f}%"
    )

    print(
        "Validation F1:",
        f"{best['f1'] * 100:.2f}%"
    )

    print()
    print(
        "Tuning complete."
    )


if __name__ == "__main__":
    main()