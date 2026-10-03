from __future__ import annotations
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class XAIIDSPipeline:
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.pipeline = None
        self.X_train = None
        self.y_train = None
        self.target = None
        self.feature_names_ = None
        self.explainer = None
        self.is_fitted = False
        self.load()

    @staticmethod
    def _target(df):
        for c in ["label", "Label", "attack", "Attack"]:
            if c in df.columns:
                return c
        raise ValueError("Target column not found. Expected label/Label/attack/Attack.")

    def prepare(self, df):
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        target = self._target(df)
        y = pd.to_numeric(df[target], errors="coerce")
        if y.isna().any():
            mapping = {"benign":0, "normal":0, "0":0, "false":0,
                       "attack":1, "malicious":1, "1":1, "true":1}
            y = df[target].astype(str).str.strip().str.lower().map(mapping)
            if y.isna().any():
                raise ValueError("Unsupported or missing target labels.")

        X = df.drop(columns=[target])
        X = X.drop(columns=[c for c in X.columns if c.lower() == "attack_cat"], errors="ignore")
        X = X.drop(columns=[c for c in X.columns if c.lower() == "id"], errors="ignore")

        return X, y.astype(int), target

    def _build(self, X, n_estimators, max_depth, learning_rate):
        numeric = X.select_dtypes(include=np.number).columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        cat = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        pre = ColumnTransformer([
            ("num", num, numeric),
            ("cat", cat, categorical)
        ])
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=1,
            gamma=0,
            subsample=0.95,
            colsample_bytree=0.95,
            reg_alpha=0.05,
            reg_lambda=1.2,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=4
        )
        return Pipeline([("preprocess", pre), ("model", model)])

    def fit(self, df, n_estimators=700, max_depth=5, learning_rate=0.035):
        X, y, target = self.prepare(df)
        self.X_train, self.y_train, self.target = X, y, target
        self.pipeline = self._build(X, n_estimators, max_depth, learning_rate)
        t0 = time.perf_counter()
        self.pipeline.fit(X, y)
        self.is_fitted = True
        self.explainer = shap.TreeExplainer(self.pipeline.named_steps["model"])
        self.feature_names_ = self.pipeline.named_steps["preprocess"].get_feature_names_out()
        self.save()
        return {"train_seconds": time.perf_counter()-t0, "rows": len(X), "features": X.shape[1]}

    def predict(self, X, threshold=0.5):
        if not self.is_fitted:
            raise RuntimeError("Model is not trained.")
        p = self.pipeline.predict_proba(X)[:, 1]
        return (p >= threshold).astype(int), p

    def evaluate(self, df, threshold=0.5):
        X, y, _ = self.prepare(df)
        pred, prob = self.predict(X, threshold)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        out = pd.DataFrame({
            "Prediction": np.where(pred==1, "ATTACK", "BENIGN"),
            "Attack Probability": np.round(prob, 5),
            "Actual": np.where(y.to_numpy()==1, "ATTACK", "BENIGN")
        })
        return {
            "accuracy": accuracy_score(y,pred),
            "precision": precision_score(y,pred,zero_division=0),
            "recall": recall_score(y,pred,zero_division=0),
            "f1": f1_score(y,pred,zero_division=0),
            "fpr": fp / max(1, fp+tn),
            "predictions": out
        }

    def explain(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model is not trained.")
        Xt = self.pipeline.named_steps["preprocess"].transform(X)
        vals = np.asarray(self.explainer.shap_values(Xt)).reshape(-1)
        names = self.feature_names_
        return (pd.DataFrame({"feature": names, "contribution": vals})
                .assign(abs=lambda d: d.contribution.abs())
                .sort_values("abs", ascending=False)
                .drop(columns="abs"))

    def quality(self, X):
        # Evaluate explanation quality on one sample
        if len(X) > 1:
            X = X.iloc[[0]].copy()
        else:
            X = X.copy()

        # 1. Explanation latency
        t0 = time.perf_counter()
        exp = self.explain(X).copy()
        latency = (time.perf_counter() - t0) * 1000.0

        if exp.empty:
            return {
                "fidelity": 0.0,
                "stability": 0.0,
                "comprehensiveness": 0.0,
                "latency_ms": latency,
                "overall": 0.0,
                "quality_score": 0.0,
            }

        exp["abs_contribution"] = exp["contribution"].abs()
        exp = exp.sort_values(
            "abs_contribution",
            ascending=False
        )

        top_k = min(10, len(exp))
        top = exp.head(top_k)

        # 2. Top-k SHAP fidelity
        try:
            _, actual_prob = self.predict(X, threshold=0.5)
            actual_prob = float(actual_prob[0])

            expected = np.asarray(
                self.explainer.expected_value
            ).reshape(-1)

            base_value = float(expected[-1])

            top_margin = (
                base_value
                + float(top["contribution"].sum())
            )

            top_margin = np.clip(top_margin, -50.0, 50.0)

            top_prob = float(
                1.0 / (1.0 + np.exp(-top_margin))
            )

            fidelity = float(
                np.clip(
                    1.0 - abs(actual_prob - top_prob),
                    0.0,
                    1.0
                )
            )

        except Exception:
            fidelity = 0.0

        # 3. Stability under 1% numerical perturbation
        try:
            X2 = X.copy().astype(object)

            for c in X.select_dtypes(
                include=np.number
            ).columns:
                v = X.iloc[0][c]

                if pd.notna(v):
                    v = float(v)

                    X2.loc[
                        X2.index[0], c
                    ] = v * 1.01 if v != 0 else 0.01

            e2 = self.explain(X2).copy()

            e1_map = exp.set_index(
                "feature"
            )["contribution"]

            e2_map = e2.set_index(
                "feature"
            )["contribution"]

            common = e1_map.index.intersection(
                e2_map.index
            )

            if len(common) > 0:
                a = e1_map.loc[common].to_numpy(
                    dtype=float
                )
                b = e2_map.loc[common].to_numpy(
                    dtype=float
                )

                norm_a = float(np.linalg.norm(a))
                norm_b = float(np.linalg.norm(b))

                if norm_a < 1e-12 and norm_b < 1e-12:
                    cosine_similarity = 1.0
                elif norm_a < 1e-12 or norm_b < 1e-12:
                    cosine_similarity = 0.0
                else:
                    cosine_similarity = float(
                        np.dot(a, b)
                        / (norm_a * norm_b)
                    )

                cosine_similarity = float(
                    np.clip(
                        (cosine_similarity + 1.0) / 2.0,
                        0.0,
                        1.0
                    )
                )

                k = min(10, len(common))

                original_top = set(
                    e1_map.loc[common]
                    .abs()
                    .sort_values(ascending=False)
                    .head(k)
                    .index
                )

                perturbed_top = set(
                    e2_map.loc[common]
                    .abs()
                    .sort_values(ascending=False)
                    .head(k)
                    .index
                )

                rank_overlap = (
                    len(
                        original_top.intersection(
                            perturbed_top
                        )
                    ) / max(1, k)
                )

                stability = float(
                    np.clip(
                        0.5 * cosine_similarity
                        + 0.5 * rank_overlap,
                        0.0,
                        1.0
                    )
                )
            else:
                stability = 0.0

        except Exception:
            stability = 0.0

        # 4. Comprehensiveness
        # Mask source features corresponding to top SHAP features
        try:
            preprocess = self.pipeline.named_steps["preprocess"]

            numeric_cols = []
            categorical_cols = []

            for name, transformer, columns in preprocess.transformers_:
                if name == "num":
                    numeric_cols.extend(list(columns))
                elif name == "cat":
                    categorical_cols.extend(list(columns))

            def source_feature(feature_name):
                feature_name = str(feature_name)

                if "__" not in feature_name:
                    return feature_name

                prefix, rest = feature_name.split(
                    "__", 1
                )

                if prefix == "num":
                    return rest

                if prefix == "cat":
                    for col in sorted(
                        categorical_cols,
                        key=len,
                        reverse=True
                    ):
                        if (
                            rest == col
                            or rest.startswith(col + "_")
                        ):
                            return col

                return rest

            source_features = []

            for feature in top["feature"].tolist():
                source = source_feature(feature)

                if source in X.columns:
                    if source not in source_features:
                        source_features.append(source)

            if source_features:
                X_mask = X.copy()

                for c in source_features:
                    X_mask[c] = X_mask[c].astype(object)
                    X_mask.loc[
                        X_mask.index[0], c
                    ] = np.nan

                base_pred, base_prob = self.predict(
                    X,
                    threshold=0.5
                )

                _, masked_prob = self.predict(
                    X_mask,
                    threshold=0.5
                )

                base_probability = float(base_prob[0])
                masked_probability = float(masked_prob[0])

                if int(base_pred[0]) == 1:
                    base_confidence = base_probability
                    masked_confidence = masked_probability
                else:
                    base_confidence = 1.0 - base_probability
                    masked_confidence = 1.0 - masked_probability

                confidence_drop = (
                    base_confidence - masked_confidence
                )

                comprehensiveness = float(
                    np.clip(
                        confidence_drop,
                        0.0,
                        1.0
                    )
                )
            else:
                comprehensiveness = 0.0

        except Exception:
            comprehensiveness = 0.0

        # 5. Latency score
        latency_component = float(
            np.exp(-latency / 100.0)
        )

        # 6. Overall explanation quality
        overall = float(
            np.clip(
                0.35 * fidelity
                + 0.35 * stability
                + 0.20 * comprehensiveness
                + 0.10 * latency_component,
                0.0,
                1.0
            )
        )

        quality_score = overall * 100.0

        return {
            "fidelity": fidelity,
            "stability": stability,
            "comprehensiveness": comprehensiveness,
            "latency_ms": latency,
            "overall": overall,
            "quality_score": quality_score,
        }

    def save(self):
        if self.pipeline is not None:
            joblib.dump(self.pipeline, self.model_dir/"xai_ids_pipeline.joblib")

    def load(self):
        path = self.model_dir/"xai_ids_pipeline.joblib"
        if path.exists():
            self.pipeline = joblib.load(path)
            self.is_fitted = True
            self.explainer = shap.TreeExplainer(self.pipeline.named_steps["model"])
            self.feature_names_ = self.pipeline.named_steps["preprocess"].get_feature_names_out()
            return True
        return False

    def summary(self):
        return {
            "model": "XGBoost",
            "explainability": "SHAP TreeExplainer",
            "rows": 0 if self.X_train is None else len(self.X_train),
            "raw_features": 0 if self.X_train is None else self.X_train.shape[1]
        }







