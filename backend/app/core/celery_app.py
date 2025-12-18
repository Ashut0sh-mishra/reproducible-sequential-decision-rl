from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "battery_rl",
    broker=settings.REDIS_BROKER,
    backend=settings.REDIS_BACKEND,
)

celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)
