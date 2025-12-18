# Structured Logging & Monitoring Guide

## Overview

This project uses **research-grade logging**, not UI dashboards. Every experiment is recorded as structured CSV/JSON files, which you inspect post-hoc in Jupyter or analyze programmatically.

---

## Components

### 1. **logging_utils.py**
Core logging module. Key classes:

- **`EpisodeLogger`**: Records step-by-step trajectories
  - `log_step()`: Log price, SoC, action, reward at each timestep
  - `save_episode()`: Write CSV + JSON files
  - Output: `logs/episode_XXXX_{agent_name}.{csv,json}`

- **`ExperimentSummary`**: Aggregate results across episodes
  - `add_episode_result()`: Record final stats per episode
  - `save_summary()`: Write `reports/summary.csv`
  - `print_summary()`: Terminal summary (mean revenue, std dev, etc.)

- **`print_step_summary()`**: Optional real-time terminal output
  ```
  [t= 123] Price: €  45.23/MWh | SoC:  75.3% | Action: CHARGE (+5.00 kW) | Cumulative Revenue: €  234.56
  ```

---

### 2. **experiment_runner.py**
Orchestrates training and logging. Key class:

- **`SimpleExperimentRunner`**: Main workhorse
  - `run_episode()`: Execute one episode, log all steps, save trajectory
  - `run_experiment()`: Run multiple agents × multiple episodes
  - Outputs: Logs + aggregate summary

- **`SimpleBaselineController`**: Example rule-based agent
  - `act(obs)`: Charge low, discharge high

---

### 3. **notebooks/analysis_template.ipynb**
Post-hoc analysis notebook. Sections:

1. Load and list episode logs (CSV files)
2. Inspect one episode (tabular view)
3. Plot trajectory (price, SoC, revenue)
4. Aggregate stats across episodes
5. Multi-agent comparison

---

## Workflow

### Step 1: Run Experiments (Terminal)

```python
from experiment_runner import SimpleExperimentRunner, SimpleBaselineController
from envs.bess_env import BessEnv
import numpy as np

# Create environment
prices = np.random.uniform(20, 150, 720)  # 30 days of hourly data
env = BessEnv(price_series=prices)

# Create agents (add your RL agent here)
baseline = SimpleBaselineController(low_threshold=50, high_threshold=100)
agents = {"baseline_rule": baseline}

# Run experiments
runner = SimpleExperimentRunner(env)
runner.run_experiment(agents, num_episodes=3, verbose=True)
```

**Output:**
- `logs/episode_0000_baseline_rule.csv`
- `logs/episode_0001_baseline_rule.csv`
- `logs/episode_0002_baseline_rule.csv`
- `reports/summary.csv` (aggregate table)
- Terminal printout with final stats

---

### Step 2: Inspect Results (Jupyter)

Open **notebooks/analysis_template.ipynb** and run cells:

1. **Load logs**: Lists all episode CSVs
2. **Inspect one**: Shows price, SoC, action, revenue (tabular)
3. **Plot trajectory**: 3-panel figure (price, SoC, cumulative revenue)
4. **Aggregate**: Summary stats by agent
5. **Compare**: Bar chart of agent performance

---

## CSV Schema

Each episode log has these columns:

| Column | Type | Example | Meaning |
|--------|------|---------|---------|
| `timestep` | int | 42 | Hour index in episode |
| `price_euro_per_mwh` | float | 67.50 | Electricity price |
| `soc_percent` | float | 45.2 | Battery state of charge (0–100%) |
| `action_kw` | float | +5.50 | Power action (charge/discharge) |
| `action_label` | str | "charge" | Discrete label for readability |
| `reward` | float | 12.34 | Revenue earned this step (€) |
| `cumulative_revenue_euro` | float | 234.56 | Total revenue so far (€) |

---

## Example: Multi-Agent Comparison

```python
# In experiment_runner.py, modify run_experiment:

from agents.train_ppo import train_rl_agent  # Your RL implementation

# Train RL agent
rl_agent = train_rl_agent(prices, num_steps=100000)

# Compare with baselines
agents = {
    "baseline_rule": SimpleBaselineController(),
    "rl_ppo": rl_agent,
    # Add MPC, random, etc.
}

runner.run_experiment(agents, num_episodes=5)
```

Result: `reports/summary.csv` with revenue for each agent, easily compared.

---

## Reproducibility

All randomness is controlled via seeds:

```python
np.random.seed(42)
env = BessEnv(price_series=prices, seed=42)
runner.run_experiment(agents, num_episodes=5)
# Results are deterministic and repeatable
```

Logs include metadata (config, seed) in JSON for audit trails.

---

## FAQ

**Q: Where are experiment logs stored?**  
A: By default, `logs/` (step-by-step CSV) and `reports/` (summary CSV).

**Q: Can I visualize results live during training?**  
A: Not recommended for PhD work. Run experiments, then analyze in Jupyter. Use `verbose=True` in `run_experiment()` for terminal summaries.

**Q: How do I add a new baseline?**  
A: Subclass or implement `act(obs)` method, pass to `run_experiment()`. See `SimpleBaselineController` for template.

**Q: What if I have multiple experiments?**  
A: Each experiment gets a unique timestamp or ID. Organize logs in subdirectories: `logs/exp_001/`, `logs/exp_002/`, etc.

---

## Next: Integrate Your RL Agent

1. Train your RL agent (PPO, DQN, etc.)
2. Wrap it in an `act(obs)` interface
3. Pass to `SimpleExperimentRunner`
4. Inspect logs in Jupyter

Example integration:
```python
class RLAgentWrapper:
    def __init__(self, policy):
        self.policy = policy
    
    def act(self, obs):
        action, _ = self.policy.predict(obs, deterministic=False)
        return action

# Use it
rl_agent = RLAgentWrapper(trained_policy)
runner.run_experiment({"ppo": rl_agent}, num_episodes=5)
```
