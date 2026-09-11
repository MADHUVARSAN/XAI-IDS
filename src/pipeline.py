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
        t0 = time.perf_counter()
        exp = self.explain(X)
        top = exp.head(min(10, len(exp)))
        total = np.abs(exp["contribution"]).sum() + 1e-9
        fidelity = float(np.clip(1 - np.abs(top["contribution"]).sum()/total, 0, 1))
        
        X2 = X.copy()
        for c in X2.select_dtypes(include=np.number).columns:
            v = X2.iloc[0][c]
            if pd.notna(v):
                X2.loc[X2.index[0], c] = v*1.0001 if v != 0 else 1e-4
        try:
            e2 = self.explain(X2).set_index("feature")
            e1 = exp.set_index("feature")
            common = e1.index.intersection(e2.index)
            a, b = e1.loc[common,"contribution"].to_numpy(), e2.loc[common,"contribution"].to_numpy()
            stability = float(np.clip(1-np.mean(np.abs(a-b))/(np.mean(np.abs(a))+1e-9),0,1))
        except Exception:
            stability = 0.0
        comprehensiveness = float(min(1, len(top)/max(1,len(exp))))
        latency = (time.perf_counter()-t0)*1000
        latency_component = 1/(1+latency/100)
        score = 100*(0.35*fidelity+0.35*stability+0.20*comprehensiveness+0.10*latency_component)
        return {"fidelity":fidelity,"stability":stability,
                "comprehensiveness":comprehensiveness,
                "latency_ms":latency,"quality_score":score}

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


