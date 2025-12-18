import numpy as np
import pandas as pd

def generate_price_series(days=90, base_price=50, seed=0):
    np.random.seed(seed)
    hours = days * 24
    t = np.arange(hours)

    daily = 10 * np.sin(2 * np.pi * (t % 24) / 24)
    weekly = 5 * np.sin(2 * np.pi * (t % (24*7)) / (24*7))
    trend = 0.01 * t
    noise = np.random.normal(0, 2, size=hours)

    spikes = np.zeros(hours)
    spike_indices = np.random.choice(hours, size=int(0.02*hours), replace=False)
    spikes[spike_indices] = np.random.choice([20, 40, -15], size=len(spike_indices))

    prices = base_price + daily + weekly + trend + noise + spikes
    prices = np.clip(prices, 0, None)

    df = pd.DataFrame({
        "hour": pd.date_range("2023-01-01", periods=hours, freq="H"),
        "price": prices
    })
    return df


if __name__ == "__main__":
    df = generate_price_series()
    df.to_csv("synthetic_prices.csv", index=False)
    print("synthetic_prices.csv generated!")
