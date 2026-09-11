Set-Location $PSScriptRoot
if (!(Test-Path ".venv")) { python -m venv .venv }
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
