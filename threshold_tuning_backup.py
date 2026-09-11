import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier


TRAIN_PATH = r"C:\Users\HP\Downloads\UNSW_NB15_training-set.csv"


df = pd.read_csv(TRAIN_PATH)

y = df["label"].astype(int)

X = df.drop(columns=["label"])

X = X.drop(
    columns=[
        c for c in X.columns
        if c.lower() in {"attack_cat", "id"}
    ],
    errors="ignore"
)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


numeric = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical = [
    c for c in X_train.columns
    if c not in numeric
]


preprocessor = ColumnTransformer(
    [
        (
            "num",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]),
            numeric
        ),
        (
            "cat",
            Pipeline([
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
            ]),
            categorical
        )
    ]
)


print()
print("=" * 70)
print("XAI-IDS THRESHOLD OPTIMIZATION")
print("=" * 70)
print()

print("Preprocessing data...")

X_train_t = preprocessor.fit_transform(
    X_train
)

X_val_t = preprocessor.transform(
    X_val
)

print(
    "Transformed validation shape:",
    X_val_t.shape
)


model = XGBClassifier(
    n_estimators=900,
    max_depth=7,
    learning_rate=0.025,
    min_child_weight=1,
    gamma=0.05,
    subsample=0.95,
    colsample_bytree=0.95,
    reg_alpha=0.01,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)


print()
print("Training Configuration C...")

model.fit(
    X_train_t,
    y_train
)

probability = model.predict_proba(
    X_val_t
)[:, 1]


thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]


results = []


for threshold in thresholds:

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        prediction,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_val,
        prediction
    )

    precision = precision_score(
        y_val,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        prediction,
        zero_division=0
    )

    fpr = fp / max(
        1,
        fp + tn
    )

    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr
        }
    )


result_df = pd.DataFrame(
    results
)


print()
print("=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)
print()


display_df = result_df.copy()

for column in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "fpr"
]:
    display_df[column] = (
        display_df[column] * 100
    ).round(2)


print(
    display_df.to_string(
        index=False
    )
)


best_accuracy = result_df.loc[
    result_df["accuracy"].idxmax()
]

best_precision = result_df.loc[
    result_df["precision"].idxmax()
]

best_f1 = result_df.loc[
    result_df["f1"].idxmax()
]


print()
print("=" * 70)
print("BEST ACCURACY THRESHOLD")
print("=" * 70)

print(
    "Threshold:",
    best_accuracy["threshold"]
)

print(
    "Accuracy:",
    f"{best_accuracy['accuracy'] * 100:.2f}%"
)

print(
    "Precision:",
    f"{best_accuracy['precision'] * 100:.2f}%"
)

print(
    "Recall:",
    f"{best_accuracy['recall'] * 100:.2f}%"
)

print(
    "F1:",
    f"{best_accuracy['f1'] * 100:.2f}%"
)

print(
    "FPR:",
    f"{best_accuracy['fpr'] * 100:.2f}%"
)


print()
print("=" * 70)
print("BEST PRECISION THRESHOLD")
print("=" * 70)

print(
    "Threshold:",
    best_precision["threshold"]
)

print(
    "Accuracy:",
    f"{best_precision['accuracy'] * 100:.2f}%"
)

print(
    "Precision:",
    f"{best_precision['precision'] * 100:.2f}%"
)

print(
    "Recall:",
    f"{best_precision['recall'] * 100:.2f}%"
)

print(
    "F1:",
    f"{best_precision['f1'] * 100:.2f}%"
)

print(
    "FPR:",
    f"{best_precision['fpr'] * 100:.2f}%"
)


print()
print("=" * 70)
print("BEST F1 THRESHOLD")
print("=" * 70)

print(
    "Threshold:",
    best_f1["threshold"]
)

print(
    "Accuracy:",
    f"{best_f1['accuracy'] * 100:.2f}%"
)

print(
    "Precision:",
    f"{best_f1['precision'] * 100:.2f}%"
)

print(
    "Recall:",
    f"{best_f1['recall'] * 100:.2f}%"
)

print(
    "F1:",
    f"{best_f1['f1'] * 100:.2f}%"
)

print(
    "FPR:",
    f"{best_f1['fpr'] * 100:.2f}%"
)

print()
print("Threshold optimization complete.")