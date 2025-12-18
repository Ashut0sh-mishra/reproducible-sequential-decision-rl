@echo off
cd /d %~dp0
echo ==========================================
echo   Starting Battery RL Full Stack Project
echo ==========================================

:: ------------ REDIS (MEMURAI) ------------
echo Checking Redis (Memurai) service...
net start Memurai >nul 2>&1
echo Redis OK.

:: ------------ CELERY WORKER ------------
echo.
echo Starting Celery Worker...
start "Celery Worker" cmd /k "cd backend && run_celery.bat"

:: ------------ FASTAPI BACKEND ------------
echo.
echo Starting FastAPI Backend on port 9001...
start "FastAPI Backend" cmd /k "cd backend && run_backend.bat"

:: ------------ FLOWER DASHBOARD ------------
echo.
echo Starting Celery Flower on port 9002...
start "Celery Flower" cmd /k "cd backend && run_flower.bat"

:: ------------ ANGULAR FRONTEND ------------
echo.
echo Starting Angular Frontend on port 5178...
start "Angular Frontend" cmd /k "cd frontend && ng serve --port 5178"

:: ------------ DONE ------------
echo.
echo ==========================================
echo  ALL SERVICES STARTED SUCCESSFULLY!
echo ------------------------------------------
echo  FRONTEND (Angular):  http://localhost:5178
echo  BACKEND (FastAPI):   http://localhost:9001
echo  FLOWER Dashboard:    http://localhost:9002
echo ==========================================
pause

