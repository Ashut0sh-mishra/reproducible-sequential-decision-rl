# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np

# class BessEnv(gym.Env):
#     metadata = {"render_modes": ["human"]}

#     def __init__(self, price_series, battery_capacity_kwh=50.0, p_max_kw=10.0, eff=0.95):
#         super().__init__()

#         self.price_series = price_series
#         self.n_steps = len(price_series)
#         self.battery_capacity = battery_capacity_kwh
#         self.p_max = p_max_kw
#         self.eff = eff

#         # Observation: [SOC, price_now, price_next, hour]
#         low = np.array([0.0, 0.0, 0.0, 0.0])
#         high = np.array([1.0, 500.0, 500.0, 23.0])
#         self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

#         self.action_space = spaces.Box(
#             low=np.array([-self.p_max]),
#             high=np.array([self.p_max]),
#             dtype=np.float32
#         )

#         self.reset()

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.t = 0
#         self.soc = 0.5
#         return self._get_obs(), {}

#     def _get_obs(self):
#         hour = self.t % 24
#         price_t = self.price_series[self.t]
#         price_next = self.price_series[min(self.t + 1, self.n_steps - 1)]
#         return np.array([self.soc, price_t, price_next, hour], dtype=np.float32)

#     def step(self, action):

#         # ✅ TERMINATION SHOULD BE CHECKED BEFORE USING self.t INDEX
#         if self.t >= self.n_steps - 1:
#             # Episode finished — return final obs safely
#             return self._get_obs(), 0.0, True, False, {}

#         # ---------- Normal step logic ----------
#         a = float(np.clip(action, -self.p_max, self.p_max))

#         if a >= 0:
#             delta_e = a * self.eff
#             cost = a * self.price_series[self.t]
#         else:
#             delta_e = a / self.eff
#             cost = a * self.price_series[self.t]

#         soc_kwh = self.soc * self.battery_capacity
#         soc_kwh_new = soc_kwh + delta_e

#         # Penalties for over/under charge
#         penalty = 0.0
#         if soc_kwh_new < 0:
#             penalty -= 100 * abs(soc_kwh_new)
#             soc_kwh_new = 0
#         if soc_kwh_new > self.battery_capacity:
#             penalty -= 100 * abs(soc_kwh_new - self.battery_capacity)
#             soc_kwh_new = self.battery_capacity

#         self.soc = soc_kwh_new / self.battery_capacity

#         degradation_penalty = 0.001 * abs(a)
#         reward = -cost - degradation_penalty + penalty

#         # Move time forward
#         self.t += 1

#         terminated = self.t >= self.n_steps - 1

#         return self._get_obs(), reward, terminated, False, {}

#     def render(self):
#         print(f"Time: {self.t}, SOC: {self.soc:.2f}")
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class BessEnv(gym.Env):
    """
    Battery Energy Storage System environment with throughput-based degradation cost.
    Observation: [SOC (0..1), price_now, price_next, hour_of_day]
    Action: continuous power in kW between -p_max..+p_max  (negative = discharge)
    Reward: revenue_from_discharge - cost_of_charge - degradation_cost - penalties
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_series,
        battery_capacity_kwh=50.0,
        p_max_kw=10.0,
        eff=0.95,
        cost_per_full_cycle=10.0,   # monetary units per full equivalent cycle (tunable)
        soc_init=0.5
    ):
        super().__init__()

        # data / parameters
        self.price_series = np.array(price_series, dtype=np.float32)
        self.n_steps = len(self.price_series)

        # battery params
        self.battery_capacity = float(battery_capacity_kwh)   # kWh
        self.p_max = float(p_max_kw)                         # kW (applied over 1-hour step -> kWh)
        self.eff = float(eff)
        self.soc_init = float(soc_init)

        # Degradation parameter (monetary cost per full cycle)
        # One full equivalent cycle throughput = 2 * battery_capacity (kWh)
        self.cost_per_full_cycle = float(cost_per_full_cycle)

        # observation & action spaces
        low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1000.0, 1000.0, 23.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.p_max], dtype=np.float32),
            high=np.array([self.p_max], dtype=np.float32),
            dtype=np.float32
        )

        # runtime state
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = float(self.soc_init)            # state of charge (0..1)
        # cumulative throughput in kWh (sum of abs energy moved)
        self.cumulative_throughput_kwh = 0.0
        # cumulative degradation cost incurred so far
        self.cumulative_degradation_cost = 0.0
        # cumulative revenue/cost (monetary)
        self.cumulative_energy_cost = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        hour = float(self.t % 24)
        price_t = float(self.price_series[self.t])
        price_next = float(self.price_series[min(self.t + 1, self.n_steps - 1)])
        return np.array([self.soc, price_t, price_next, hour], dtype=np.float32)

    def step(self, action):
        """
        action: continuous scalar (kW). Positive -> charge, Negative -> discharge.
        We treat each step as 1 hour for simplicity so energy (kWh) ~= power(kW)*1h.
        """

        # If episode ended previously (safety)
        if self.t >= self.n_steps - 1:
            return self._get_obs(), 0.0, True, False, {}

        # Clamp action to allowed power range
        a = float(np.clip(action, -self.p_max, self.p_max))

        # Energy moved in this 1-hour step (kWh)
        # For charging: energy added to battery = power * eff
        # For discharging: energy removed (kWh) = power / eff  (since a is negative for discharge)
        if a >= 0.0:
            delta_e_kwh = a * self.eff      # energy stored in battery (kWh)
            energy_transacted_kwh = a       # energy bought from grid (kWh)
        else:
            delta_e_kwh = a / self.eff      # negative value (kWh removed from battery)
            energy_transacted_kwh = a       # negative => energy sold to grid

        # Update SOC (in kWh)
        soc_kwh = self.soc * self.battery_capacity
        soc_kwh_new = soc_kwh + delta_e_kwh

        # Bound SOC to [0, capacity]
        penalty = 0.0
        if soc_kwh_new < 0.0:
            penalty -= 100.0 * abs(soc_kwh_new)   # large penalty for violations
            soc_kwh_new = 0.0
        if soc_kwh_new > self.battery_capacity:
            penalty -= 100.0 * (soc_kwh_new - self.battery_capacity)
            soc_kwh_new = self.battery_capacity

        # Update SOC (0..1)
        self.soc = float(soc_kwh_new / self.battery_capacity)

        # Economic term: cost (if charging) or revenue (if discharging)
        price_t = float(self.price_series[self.t])
        # energy_transacted_kwh positive means buying (cost), negative means selling (revenue)
        energy_cost = energy_transacted_kwh * price_t   # could be negative (revenue)
        # We will use reward = -energy_cost - degradation - penalties (so agent maximizes profit)
        self.cumulative_energy_cost += energy_cost

        # --- Degradation calculation (throughput-based) ---
        throughput_added = abs(delta_e_kwh)   # absolute energy that passed through battery (kWh)
        self.cumulative_throughput_kwh += throughput_added

        # fraction of a full cycle this step contributes:
        # full_cycle_throughput = 2 * capacity (kWh)
        frac_full_cycle = throughput_added / (2.0 * self.battery_capacity)
        degradation_cost_step = frac_full_cycle * self.cost_per_full_cycle
        self.cumulative_degradation_cost += degradation_cost_step

        # small instantaneous smoothing penalty (optional)
        smoothing_penalty = 0.0

        # Total reward: negative cost (so maximizing reward => maximizing revenue),
        # subtract degradation cost and boundary penalties
        reward = -energy_cost - degradation_cost_step + penalty - smoothing_penalty

        # advance time
        self.t += 1
        terminated = self.t >= self.n_steps - 1

        # Add informative info dictionary (useful for evaluation & logging)
        info = {
            "energy_cost_step": float(energy_cost),
            "degradation_cost_step": float(degradation_cost_step),
            "cum_throughput_kwh": float(self.cumulative_throughput_kwh),
            "cum_degradation_cost": float(self.cumulative_degradation_cost),
            "soc": float(self.soc)
        }

        return self._get_obs(), float(reward), bool(terminated), False, info

    def render(self):
        print(f"Time: {self.t}, SOC: {self.soc:.3f}, CumThroughput(kWh): {self.cumulative_throughput_kwh:.2f}, CumDegrCost: {self.cumulative_degradation_cost:.3f}")
