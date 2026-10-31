@echo off
echo ===================================================
echo Starting EnterpriseDoc AI (Backend + Frontend)
echo ===================================================

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "EnterpriseDoc AI - Backend" cmd /k ".\venv\Scripts\python.exe run_backend.py"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Streamlit Frontend on http://localhost:8501 ...
start "EnterpriseDoc AI - Frontend" cmd /k ".\venv\Scripts\streamlit.exe run frontend\app.py --server.port 8501"

echo.
echo EnterpriseDoc AI is launching!
echo Backend:  http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)
echo Frontend: http://localhost:8501
echo ===================================================
pause
