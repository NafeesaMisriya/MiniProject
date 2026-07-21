@echo off
echo ===================================================
echo Starting ModelGuard AI Risk Engine Backend...
echo ===================================================
echo.
echo Opening web application at http://127.0.0.1:8000...
start http://127.0.0.1:8000
echo.
python -m uvicorn main:app --port 8000 --host 127.0.0.1
pause
