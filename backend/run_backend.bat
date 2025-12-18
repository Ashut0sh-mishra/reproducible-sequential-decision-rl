@echo off
cd /d %~dp0
set PYTHONPATH=%CD%

echo Starting FastAPI backend on port 9001...
uvicorn app.main:app --host 0.0.0.0 --port 9001 --reload
pause
