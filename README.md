# XAI-IDS — Working High-Performance Prototype

This build follows the supplied project presentation:
**Detect → Explain → Measure → Validate → SOC Support.**

## Accuracy goal

The XGBoost configuration is deliberately stronger than a basic baseline:
- 700 boosting rounds by default
- low learning rate
- tuned depth
- high row/feature subsampling
- regularization
- histogram training

The dashboard calculates metrics from the **actual uploaded test CSV**. It does not fabricate 99.5%.

If your current implementation already gives ~98.9%, this build is intended as a clean optimization base. Whether it reaches 99.5% depends on the exact train/test files and preprocessing.

## Run on Windows PowerShell

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Upload:
1. `UNSW_NB15_training-set.csv`
2. `UNSW_NB15_testing-set.csv`

Then train and evaluate.

## Important dataset note

The `attack_cat` field is excluded if it is not the binary target in the future extension. Do not use target-derived columns as predictors when claiming generalization performance.
