@echo off
cd /d "%~dp0"

call venv\Scripts\activate.bat

start "" http://localhost:8501

python -m streamlit run src\app.py