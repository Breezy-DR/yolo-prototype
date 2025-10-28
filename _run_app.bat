@echo off
powershell -NoExit -Command "& { .\venv\Scripts\Activate.ps1; streamlit run main.py }"
