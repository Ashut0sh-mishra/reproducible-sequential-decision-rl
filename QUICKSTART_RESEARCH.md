# Quick Start: Running PhD-Ready Experiments

## TL;DR

```bash
# 1. Set up Python environment
cd backend
pip install -r requirements.txt

# 2. Start backend
python -m uvicorn app.main:app --reload

# 3. Run experiment via curl
curl -X POST http://localhost:8000/experiments/run-experiment \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "rule_based",
    "model_params": {"low_threshold": 50, "high_threshold": 100},
    "environment_params": {"battery_capacity_kwh": 50, "p_max_kw": 10},
    "seed": 42,
    "num_episodes": 3
  }'

# 4. Analyze results in Jupyter
jupyter notebook notebooks/

# 5. Load results
import pandas as pd
df = pd.read_csv("experiments/exp_abc123/exp_abc123_summary.csv")
print(df)
```

---

## Step-by-Step Guide

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Required packages:
- fastapi, uvicorn (API server)
- numpy, pandas (data handling)
- gymnasium, torch (ML dependencies)
- pydantic (config validation)

### 2. Start Backend Server

```bash
python -m uvicorn app.main:app --reload
```

Output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 3. Run an Experiment

#### Option A: Via curl
```bash
curl -X POST http://localhost:8000/experiments/run-experiment \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "rule_based",
    "model_params": {
      "low_threshold": 50.0,
      "high_threshold": 100.0,
      "p_max": 10.0
    },
    "environment_params": {
      "battery_capacity_kwh": 50.0,
      "p_max_kw": 10.0,
      "eff": 0.95,
      "cost_per_full_cycle": 10.0
    },
    "seed": 42,
    "num_episodes": 3
  }'
```

Response:
```json
{
  "experiment_id": "a1b2c3d4",
  "status": "completed",
  "artifacts": {
    "config": "experiments/a1b2c3d4/exp_a1b2c3d4_config.json",
    "metrics": "experiments/a1b2c3d4/exp_a1b2c3d4_metrics.json",
    "summary": "experiments/a1b2c3d4/exp_a1b2c3d4_summary.csv",
    "logs": "experiments/a1b2c3d4/exp_a1b2c3d4_logs.csv"
  },
  "summary_stats": {
    "mean_reward": 1345.67,
    "std_reward": 111.11,
    "min_reward": 1234.56,
    "max_reward": 1456.78
  }
}
```

#### Option B: Via Python
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/experiments/run-experiment",
    json={
        "model_name": "rule_based",
        "model_params": {"low_threshold": 50, "high_threshold": 100},
        "environment_params": {"battery_capacity_kwh": 50},
        "seed": 42,
        "num_episodes": 3,
    }
)

result = response.json()
print(json.dumps(result, indent=2))
```

### 4. Inspect Results

#### Option A: View files directly
```bash
# Config (reproducibility audit trail)
cat experiments/a1b2c3d4/exp_a1b2c3d4_config.json

# Summary (episode-level results)
cat experiments/a1b2c3d4/exp_a1b2c3d4_summary.csv

# Logs (step-by-step trajectory)
head -20 experiments/a1b2c3d4/exp_a1b2c3d4_logs.csv

# Metrics (aggregate statistics)
cat experiments/a1b2c3d4/exp_a1b2c3d4_metrics.json
```

#### Option B: Via API
```bash
# Get experiment summary
curl http://localhost:8000/experiments/experiment/a1b2c3d4

# Get step-by-step logs
curl http://localhost:8000/experiments/experiment/a1b2c3d4/logs?limit=100

# List all experiments
curl http://localhost:8000/experiments/list
```

#### Option C: Via Python/Jupyter
```python
import pandas as pd
import json

# Load summary
df_summary = pd.read_csv("experiments/a1b2c3d4/exp_a1b2c3d4_summary.csv")
print(df_summary)

# Load logs
df_logs = pd.read_csv("experiments/a1b2c3d4/exp_a1b2c3d4_logs.csv")
print(f"Total steps: {len(df_logs)}")
print(f"Cumulative reward: {df_logs['cumulative_reward'].iloc[-1]:.2f}")

# Load config
with open("experiments/a1b2c3d4/exp_a1b2c3d4_config.json") as f:
    config = json.load(f)
print(f"Seed: {config['seed']}, Model: {config['model_name']}")
```

### 5. Run Multiple Models (Comparison)

```python
# Run rule-based
result_rule = requests.post(
    "http://localhost:8000/experiments/run-experiment",
    json={
        "model_name": "rule_based",
        "seed": 42,
        "num_episodes": 3,
    }
).json()

# Run random
result_random = requests.post(
    "http://localhost:8000/experiments/run-experiment",
    json={
        "model_name": "random",
        "seed": 42,
        "num_episodes": 3,
    }
).json()

# Load and compare
from app.core.result_inspector import ExperimentResultsInspector

inspector = ExperimentResultsInspector()
df = inspector.compare_experiments([
    result_rule["experiment_id"],
    result_random["experiment_id"],
])
print(df[["model", "mean_reward", "std_reward"]])
```

### 6. Jupyter Analysis

Create `notebooks/analysis.ipynb` and load results:

```python
import pandas as pd
import json
from pathlib import Path

# List all experiments
exp_dir = Path("../experiments")
experiments = [d.name for d in exp_dir.iterdir() if d.is_dir()]
print(f"Found {len(experiments)} experiments:")
for exp_id in experiments:
    print(f"  - {exp_id}")

# Load a specific experiment
exp_id = experiments[0]
df_summary = pd.read_csv(f"../experiments/{exp_id}/exp_{exp_id}_summary.csv")
df_logs = pd.read_csv(f"../experiments/{exp_id}/exp_{exp_id}_logs.csv")

print(f"\nExperiment {exp_id}:")
print(df_summary)
print(f"\nMean reward: {df_summary['total_reward'].mean():.2f}")
print(f"Std dev: {df_summary['total_reward'].std():.2f}")

# Plot cumulative reward
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Episode rewards
axes[0].bar(df_summary['episode'], df_summary['total_reward'])
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward (€)')
axes[0].set_title('Episode Rewards')

# Cumulative reward trajectory (first episode)
ep0_logs = df_logs[df_logs['episode'] == 0]
axes[1].plot(ep0_logs['step'], ep0_logs['cumulative_reward'])
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Cumulative Reward (€)')
axes[1].set_title('Episode 0: Cumulative Reward Trajectory')

plt.tight_layout()
plt.show()
```

---

## Available Models

| Model | Params | Use Case |
|-------|--------|----------|
| `rule_based` | `low_threshold`, `high_threshold`, `p_max` | Simple baseline |
| `random` | `p_max`, `seed` | Sanity check (should be worst) |
| `greedy` | `p_max`, `buy_threshold`, `sell_threshold` | Myopic optimization |
| `dqn` | (TBD) | Neural network policy |

---

## Reproducibility Check

Verify that identical configs produce identical results:

```python
import requests

# Run experiment twice with same seed
exp1 = requests.post(
    "http://localhost:8000/experiments/run-experiment",
    json={
        "model_name": "rule_based",
        "seed": 42,
        "num_episodes": 3,
    }
).json()

exp2 = requests.post(
    "http://localhost:8000/experiments/run-experiment",
    json={
        "model_name": "rule_based",
        "seed": 42,
        "num_episodes": 3,
    }
).json()

# Compare
assert exp1["summary_stats"]["mean_reward"] == exp2["summary_stats"]["mean_reward"]
print("✓ Reproducibility verified: Same seed → identical results")
```

---

## Troubleshooting

### Backend won't start
```bash
# Check Python version (need 3.8+)
python --version

# Check dependencies
pip install -r requirements.txt

# Try verbose mode
python -m uvicorn app.main:app --reload --log-level debug
```

### Experiments fail
```bash
# Check environment path
ls envs/bess_env.py

# Check model name
curl http://localhost:8000/experiments/list

# Check seed is integer
"seed": 42  # ✓ Good
"seed": "42"  # ✗ Bad
```

### Results not found
```bash
# Check experiments directory
ls experiments/

# Check file permissions
ls -la experiments/*/

# Try loading via Python
import json
config = json.load(open("experiments/exp_abc123/exp_abc123_config.json"))
```

---

## Next Steps

1. ✅ Run first experiment with rule-based model
2. ✅ Compare multiple models
3. ⏳ Train and integrate DQN/PPO model
4. ⏳ Generate thesis results (multiple seeds, statistical tests)
5. ⏳ Archive all artifacts for PhD submission

---

## Key Files

| File | Purpose |
|------|---------|
| `backend/app/core/experiment_artifacts.py` | Artifact schema |
| `backend/app/core/research_runner.py` | Experiment execution |
| `backend/app/agents/research_models.py` | Model definitions |
| `backend/app/core/result_inspector.py` | Result loading |
| `backend/app/api/experiments.py` | API endpoints |
| `experiments/` | All experiment artifacts (CSV/JSON) |
| `RESEARCH_ARCHITECTURE.md` | Technical design |
| `REPRODUCIBILITY_GUIDE.md` | How to verify results |

---

**Status:** Backend is research-ready. Run experiments → save artifacts → analyze in Jupyter.
