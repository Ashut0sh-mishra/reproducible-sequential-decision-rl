from celery import Celery
from app.core.config import settings
import pandas as pd
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from agents.train_ppo import train_model


celery_app = Celery(
    "tasks",
    broker=settings.REDIS_BROKER,
    backend=settings.REDIS_BACKEND
)


@celery_app.task(bind=True)
def train_battery_rl(self, degradation_cost: float, timesteps: int):
    """
    Celery task with LIVE progress updates
    """

    # Load price data
    price_path = "data/synthetic_prices.csv"
    prices = pd.read_csv(price_path)["price"].values.tolist()

    # ✅ INITIAL STATE (NO step VARIABLE YET)
    self.update_state(
        state="PROGRESS",
        meta={
            "step": 0,
            "total": timesteps,
            "progress": 0
        }
    )

    # ✅ PROGRESS CALLBACK (step EXISTS HERE)
    def progress_callback(step: int):
        progress = max(1, int(step / timesteps * 100))

        self.update_state(
            state="PROGRESS",
            meta={
                "step": step,
                "total": timesteps,
                "progress": progress
            }
        )

    # 🔥 TRAINING
    model_path, metrics = train_model(
        prices=prices,
        degradation_cost=degradation_cost,
        timesteps=timesteps,
        progress_callback=progress_callback
    )

    return {
        "message": "Training completed",
        "model_path": model_path,
        "metrics": metrics,
        "progress": 100
    }
