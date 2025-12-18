@echo off
cd /d %~dp0
set PYTHONPATH=%CD%
echo Starting Celery Worker...
celery -A app.workers.train_task.celery_app worker --loglevel=info
pause
