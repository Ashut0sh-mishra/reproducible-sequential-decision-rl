@echo off
cd /d %~dp0
set PYTHONPATH=%CD%\..
echo Starting Celery Flower on port 9002...
celery -A backend.app.workers.train_task.celery_app flower --port=9002
pause
