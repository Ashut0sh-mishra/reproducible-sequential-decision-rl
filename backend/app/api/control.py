from fastapi import APIRouter
from app.core.celery_app import celery_app
import subprocess
import requests

router = APIRouter(prefix="/control", tags=["Control"])


# =========================
# SYSTEM STATUS
# =========================
@router.get("/status")
def system_status():
    status = {
        "fastapi": True,
        "celery": False,
        "workers": [],
        "flower": False
    }

    # ---- CELERY WORKER STATUS (REAL) ----
    try:
        responses = celery_app.control.ping(timeout=1)
        if responses:
            status["celery"] = True
            status["workers"] = [list(r.keys())[0] for r in responses]
    except Exception:
        pass

    # ---- FLOWER STATUS (HTTP HEALTH CHECK) ----
    try:
        r = requests.get("http://localhost:5555", timeout=1)
        status["flower"] = (r.status_code == 200)
    except Exception:
        pass

    return status


# =========================
# START CELERY WORKER
# =========================
@router.post("/start")
def start_celery():
    responses = celery_app.control.ping(timeout=1)
    if responses:
        return {"celery": "already running"}

    subprocess.Popen(
        ["celery", "-A", "app.workers.train_task", "worker", "--loglevel=info"],
        shell=True
    )
    return {"celery": "started"}


# =========================
# STOP CELERY WORKER
# =========================
@router.post("/stop")
def stop_celery():
    responses = celery_app.control.ping(timeout=1)

    if not responses:
        return {"celery": "already stopped"}

    celery_app.control.broadcast("shutdown")
    return {"celery": "stopped"}


# =========================
# CELERY TASKS INFO
# =========================
@router.get("/tasks")
def celery_tasks():
    inspect = celery_app.control.inspect()

    if not inspect:
        return {
            "active": {},
            "scheduled": {},
            "reserved": {}
        }

    return {
        "active": inspect.active() or {},
        "scheduled": inspect.scheduled() or {},
        "reserved": inspect.reserved() or {}
    }
