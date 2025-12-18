import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.bess_env import BessEnv
import uuid
import os

def train_model(prices, degradation_cost, timesteps):

    def make_env():
        return BessEnv(
            price_series=np.array(prices),
            cost_per_full_cycle=degradation_cost
        )

    env = make_vec_env(make_env, n_envs=4)

    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)

    model_id = str(uuid.uuid4())
    model_path = f"results/model_{model_id}.zip"
    os.makedirs("results", exist_ok=True)
    model.save(model_path)

    return model_path, {"timesteps": timesteps, "degradation_cost": degradation_cost}
