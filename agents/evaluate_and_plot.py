import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from envs.bess_env import BessEnv

prices = pd.read_csv("data/synthetic_prices.csv")["price"].values

def eval_model(model_path, capacity=50.0, cost_per_cycle=10.0):
    env = BessEnv(price_series=prices, battery_capacity_kwh=capacity, cost_per_full_cycle=cost_per_cycle)
    model = PPO.load(model_path)
    obs, _ = env.reset()
    done = False

    actions = []
    socs = []
    prices_list = []
    energy_costs = []
    degr_costs = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        actions.append(float(action))
        socs.append(info["soc"])
        prices_list.append(env.price_series[env.t-1])   # price used at that step
        energy_costs.append(info["energy_cost_step"])
        degr_costs.append(info["degradation_cost_step"])

    total_energy_cost = env.cumulative_energy_cost
    total_degr_cost = env.cumulative_degradation_cost
    net_profit = -total_energy_cost - total_degr_cost

    # Plot SOC, price, action
    t = np.arange(len(socs))
    plt.figure(figsize=(12,6))
    plt.subplot(3,1,1)
    plt.plot(t, prices_list); plt.ylabel("Price")
    plt.subplot(3,1,2)
    plt.plot(t, socs); plt.ylabel("SOC")
    plt.subplot(3,1,3)
    plt.plot(t, actions); plt.ylabel("Action kW")
    plt.tight_layout()
    plt.show()

    print("total_energy_cost:", total_energy_cost)
    print("total_degradation_cost:", total_degr_cost)
    print("net_profit:", net_profit)

if __name__ == "__main__":
    eval_model("ppo_bess_model", capacity=50.0, cost_per_cycle=10.0)
