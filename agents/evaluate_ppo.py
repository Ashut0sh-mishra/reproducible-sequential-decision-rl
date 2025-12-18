import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from envs.bess_env import BessEnv

prices = pd.read_csv("data/synthetic_prices.csv")["price"].values
env = BessEnv(price_series=prices)

model = PPO.load("ppo_bess_model")

obs, _ = env.reset()
total_reward = 0
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

print("Final reward:", total_reward)
print("Final SOC:", env.soc)
