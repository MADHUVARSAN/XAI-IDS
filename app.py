import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

from src.pipeline import XAIIDSPipeline

st.set_page_config(page_title="XAI-IDS | Trust-Aware SOC", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1450px;}
.metric-card {padding: 14px; border: 1px solid rgba(255,255,255,.12); border-radius: 12px;}
.small {opacity:.72; font-size:.85rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    return XAIIDSPipeline()

if "engine" not in st.session_state:
    st.session_state.engine = get_engine()
if "feedback" not in st.session_state:
    st.session_state.feedback = []

engine = st.session_state.engine

st.title("🛡️ XAI-IDS")
st.caption("Trust-Aware Explainable Network Intrusion Detection System • DETECT • EXPLAIN • MEASURE • VALIDATE")

with st.sidebar:
    st.header("Data & Model")
    train_file = st.file_uploader("UNSW-NB15 training CSV", type="csv")
    test_file = st.file_uploader("UNSW-NB15 testing CSV", type="csv")
    st.divider()
    st.subheader("Model tuning")
    n_estimators = st.slider("Boosting rounds", 200, 1200, 700, 50)
    max_depth = st.slider("Max depth", 3, 10, 5)
    learning_rate = st.slider("Learning rate", 0.01, 0.20, 0.035, 0.005)
    threshold = st.slider("Attack threshold", 0.10, 0.90, 0.50, 0.01)
    train_btn = st.button("🚀 Train / Retrain Model", type="primary", width="stretch")

if train_btn:
    if train_file is None:
        st.error("Upload the real UNSW-NB15 training CSV first.")
    else:
        df = pd.read_csv(train_file)
        try:
            with st.spinner("Training optimized XGBoost pipeline..."):
                info = engine.fit(df, n_estimators=n_estimators,
                                  max_depth=max_depth,
                                  learning_rate=learning_rate)
            st.success(f"Training complete in {info['train_seconds']:.2f}s • {info['rows']:,} rows • {info['features']} raw features")
        except Exception as e:
            st.exception(e)

tabs = st.tabs(["🚨 Detection", "🔍 Explain", "📊 Quality", "👤 Analyst Audit"])

with tabs[0]:
    st.subheader("Detection Posture")
    if not engine.is_fitted:
        st.info("Upload training data and train the model.")
    else:
        if test_file is not None:
            test_df = pd.read_csv(test_file)
            try:
                result = engine.evaluate(test_df, threshold=threshold)
                c = st.columns(5)
                c[0].metric("Accuracy", f"{result['accuracy']*100:.2f}%")
                c[1].metric("Precision", f"{result['precision']*100:.2f}%")
                c[2].metric("Recall", f"{result['recall']*100:.2f}%")
                c[3].metric("F1", f"{result['f1']*100:.2f}%")
                c[4].metric("FPR", f"{result['fpr']*100:.2f}%")
                st.caption(f"Threshold = {threshold:.2f}. Metrics are calculated from the uploaded unseen test set.")
                st.dataframe(result["predictions"].head(200), width="stretch")
            except Exception as e:
                st.exception(e)
        else:
            st.warning("Upload the testing CSV to calculate genuine test-set metrics.")
            st.write(engine.summary())

with tabs[1]:
    st.subheader("SHAP Local Explanation")
    if not engine.is_fitted:
        st.info("Train the model first.")
    else:
        idx = st.number_input("Training record index", 0, max(0, engine.X_train.shape[0]-1), 0)
        row = engine.X_train.iloc[[int(idx)]]
        pred, prob = engine.predict(row, threshold)
        c1, c2 = st.columns(2)
        c1.metric("Prediction", "ATTACK" if pred[0] else "BENIGN")
        c2.metric("Attack probability", f"{prob[0]*100:.2f}%")
        if st.button("Generate SHAP explanation", type="primary"):
            try:
                exp = engine.explain(row)
                st.session_state.last_exp = exp
                st.dataframe(exp.head(20), width="stretch")
                st.bar_chart(exp.head(12).set_index("feature")["contribution"])
            except Exception as e:
                st.exception(e)

with tabs[2]:
    st.subheader("Explanation Quality")
    if not engine.is_fitted:
        st.info("Train the model first.")
    else:
        idx = st.number_input("Quality record index", 0, max(0, engine.X_train.shape[0]-1), 0, key="quality_idx")
        row = engine.X_train.iloc[[int(idx)]]
        if st.button("Measure explanation quality", type="primary"):
            try:
                metrics = engine.quality(row)
                cols = st.columns(5)
                cols[0].metric("Fidelity", f"{metrics['fidelity']*100:.1f}%")
                cols[1].metric("Stability", f"{metrics['stability']*100:.1f}%")
                cols[2].metric("Comprehensiveness", f"{metrics['comprehensiveness']*100:.1f}%")
                cols[3].metric("Latency", f"{metrics['latency_ms']:.1f} ms")
                cols[4].metric("Quality", f"{metrics['quality_score']:.1f}/100")
                st.progress(float(metrics["quality_score"]/100))
            except Exception as e:
                st.exception(e)

with tabs[3]:
    st.subheader("Human-in-the-Loop Analyst Validation")
    if "last_exp" in st.session_state:
        st.dataframe(st.session_state.last_exp.head(12), width="stretch")
    judgement = st.radio("Analyst judgement", ["Useful", "Unclear", "Incorrect"], horizontal=True)
    note = st.text_area("Audit note")
    if st.button("Save analyst feedback"):
        st.session_state.feedback.append({
            "time": pd.Timestamp.now().isoformat(timespec="seconds"),
            "judgement": judgement,
            "note": note
        })
        st.success("Feedback added to audit trail.")
    if st.session_state.feedback:
        st.dataframe(pd.DataFrame(st.session_state.feedback), width="stretch")

st.divider()
st.caption("XAI-IDS prototype: real-data detection, SHAP explanation, quality measurement and analyst validation. It is a detection/decision-support system, not an automatic attack-prevention system.")

