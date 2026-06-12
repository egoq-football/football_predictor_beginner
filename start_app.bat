@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
streamlit run app.py --server.address 0.0.0.0
pause
