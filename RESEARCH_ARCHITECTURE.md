# PhD Research Architecture: Backend Refactoring

## Overview

This document explains the refactored backend architecture for research-grade reproducibility and evaluation.

**Core Principle:** All research artifacts (results) are saved to disk BEFORE the UI sees them. The UI reads from files only.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Environment (BessEnv) + Models (Rule, RL, etc.)            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ResearchExperimentRunner                                     │
│  - Sets reproducible seeds (numpy, random, torch)           │
│  - Runs episodes with each model                            │
│  - Logs EVERY step to memory                                │
│  - Computes aggregate statistics                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼  SAVES to disk
┌─────────────────────────────────────────────────────────────┐
│ Experiment Artifacts (PRIMARY RESEARCH OUTPUT)              │
│  exp_<id>_config.json     - Reproducibility audit trail     │
│  exp_<id>_metrics.json    - Aggregate statistics            │
│  exp_<id>_summary.csv     - Episode-level results           │
│  exp_<id>_logs.csv        - Step-by-step trajectories       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼  READS from disk
┌─────────────────────────────────────────────────────────────┐
│ UI / Dashboards / Inspection Tools                          │
│  - Display saved results                                    │
│  - Configure experiments                                    │
│  - Trigger runs (via API)                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. **app/core/experiment_artifacts.py**
Defines standardized artifact schema. Every experiment produces 4 files.

**Files Created:**
- `exp_<id>_config.json` – Reproducibility ground truth
  ```json
  {
    "experiment_id": "a1b2c3d4",
    "timestamp": "2025-12-17T10:30:00",
    "seed": 42,
    "model_name": "rule_based",
    "model_params": {"low_threshold": 50, "high_threshold": 100},
    "num_episodes": 3,
    "num_steps_per_episode": 720
  }
  ```

- `exp_<id>_metrics.json` – Aggregate statistics
  ```json
  {
    "episodes": [
      {"episode": 0, "total_reward": 1234.56, "num_steps": 720},
      {"episode": 1, "total_reward": 1456.78, "num_steps": 720}
    ],
    "statistics": {
      "mean_reward": 1345.67,
      "std_reward": 111.11,
      "min_reward": 1234.56,
      "max_reward": 1456.78
    }
  }
  ```

- `exp_<id>_summary.csv` – Episode summaries (easy for pandas)
  ```csv
  episode,total_reward,num_steps,avg_reward_per_step,model,seed
  0,1234.56,720,1.71,rule_based,42
  1,1456.78,720,2.02,rule_based,42
  ```

- `exp_<id>_logs.csv` – Complete step-by-step audit trail
  ```csv
  episode,step,obs_0,obs_1,action,reward,cumulative_reward,...
  0,0,0.50,67.50,5.00,12.34,12.34,...
  0,1,0.55,72.30,-5.00,8.90,21.24,...
  ```

**Usage:**
```python
from app.core.experiment_artifacts import ExperimentConfig, ExperimentArtifactManager

manager = ExperimentArtifactManager("experiments")
artifacts = manager.save_all_artifacts(
    experiment_id="exp_123",
    config=config,
    metrics=metrics,
    summary=episode_summaries,
    logs=step_logs,
)
# Returns: {"config": "...", "metrics": "...", "summary": "...", "logs": "..."}
```

---

### 2. **app/core/research_runner.py**
The experiment execution engine. Ensures reproducibility.

**Responsibilities:**
- Set ALL random seeds (numpy, random, torch)
- Run episodes with consistent state
- Log every step
- Compute aggregate metrics
- Save all artifacts via `ExperimentArtifactManager`

**Key Method:**
```python
runner = ResearchExperimentRunner(env_factory, model_factory)
result = runner.run_experiment(
    config=config,
    model_name="rule_based",
    model_params={...},
    environment_params={...},
    num_episodes=3,
)
# Returns: Dict with artifact paths + summary stats
```

**Guarantees:**
- ✅ Deterministic: Same seed → same results
- ✅ Auditable: Every step logged
- ✅ Reproducible: Seeds saved in config
- ✅ Offline: All results on disk before API response

---

### 3. **app/agents/research_models.py**
Unified model interface. All models use same `act(obs) -> action` contract.

**Available Models:**
- `RuleBasedModel` – Threshold-based arbitrage
- `RandomModel` – Uniform random baseline
- `GreedyModel` – Myopic profit maximization
- `DummyRLModel` – Placeholder for RL (DQN)

**Example:**
```python
from app.agents.research_models import model_factory

model = model_factory("rule_based", params={
    "low_threshold": 50,
    "high_threshold": 100,
    "p_max": 10.0,
})
action = model.act(observation)
```

---

### 4. **app/core/result_inspector.py**
Read-only interface for accessing saved results. NO computation.

**Methods:**
- `get_experiment_summary()` – Load from disk
- `compare_experiments()` – Side-by-side comparison
- `load_summary_table()` – Episode-level results
- `load_logs_table()` – Step-by-step audit trail
- `list_all_experiments()` – Directory listing

**Example:**
```python
inspector = ExperimentResultsInspector("experiments")
df_summary = inspector.load_summary_table("exp_123")
df_comparison = inspector.compare_experiments(["exp_123", "exp_456"])
```

---

### 5. **app/api/experiments.py (Refactored)**
API endpoints now save artifacts BEFORE returning responses.

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/experiments/run-experiment` | POST | Trigger experiment (saves artifacts) |
| `/experiments/experiment/{id}` | GET | Load results from disk |
| `/experiments/experiment/{id}/logs` | GET | Load step-by-step logs from disk |
| `/experiments/list` | GET | List all experiments |

**Key Change:**
```python
# OLD (bad): Returns task_id, results computed later
result = train_model_async(config)
return {"task_id": result.id}

# NEW (good): Results saved, paths returned
result = runner.run_experiment(config)
return result  # {"artifacts": {...}, "summary_stats": {...}}
```

---

## Reproducibility Guarantees

### 1. Fixed Seeds
```python
ResearchExperimentRunner.set_seeds(42)  # Sets numpy, random, torch
```

### 2. Config Audit Trail
```json
{
  "seed": 42,
  "model_name": "rule_based",
  "model_params": {...},
  "environment_params": {...},
  "timestamp": "2025-12-17T10:30:00"
}
```

### 3. Step-by-Step Logs
Every action, reward, and observation saved to CSV.

### 4. Deterministic Execution
Same config + seed → identical results (verified offline).

---

## UI Role (Limited & Safe)

The UI is NOT where research happens. It:
- ✅ Displays saved results from disk
- ✅ Allows experiment configuration
- ✅ Triggers experiment runs
- ❌ Does NOT compute metrics
- ❌ Does NOT store results in memory
- ❌ Does NOT claim real-time learning

**Example UI Flow:**
```
1. User selects: model=rule_based, seed=42, episodes=3
2. UI calls: POST /experiments/run-experiment
3. Backend: Runs experiment, saves artifacts to disk
4. Backend returns: {"artifacts": {...}}
5. UI: Loads results from disk, displays tables/plots
6. User can download CSV files directly
```

---

## Analysis Workflow (PhD-Ready)

### Step 1: Run Experiments
```bash
# Via API
curl -X POST http://localhost:8000/experiments/run-experiment \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "rule_based",
    "model_params": {"low_threshold": 50},
    "num_episodes": 5,
    "seed": 42
  }'
```

### Step 2: Inspect Results (Jupyter)
```python
import pandas as pd
from app.core.result_inspector import ExperimentResultsInspector

inspector = ExperimentResultsInspector()

# Load results
summary = inspector.load_summary_table("exp_abc123")
logs = inspector.load_logs_table("exp_abc123")

# Analyze
print(summary.describe())
print(f"Mean reward: {summary['total_reward'].mean():.2f}")
```

### Step 3: Compare Models (Jupyter)
```python
df = inspector.compare_experiments([
    "exp_rule_based_1",
    "exp_rule_based_2",
    "exp_ppo_1",
])
print(df[["model", "seed", "mean_reward", "std_reward"]])
```

---

## Testing Reproducibility

Verify that identical configurations produce identical results:

```python
from app.core.research_runner import ResearchExperimentRunner

runner = ResearchExperimentRunner(env_factory, model_factory)
config1 = create_research_experiment_config(seed=42, ...)
config2 = create_research_experiment_config(seed=42, ...)

result1 = runner.run_experiment(config1)
result2 = runner.run_experiment(config2)

assert result1["summary_stats"]["mean_reward"] == result2["summary_stats"]["mean_reward"]
print("✓ Reproducibility verified")
```

---

## File Structure

```
d:\battery_rl_project\
├── backend/
│   └── app/
│       ├── core/
│       │   ├── experiment_artifacts.py    ← Artifact schema
│       │   ├── research_runner.py         ← Main execution engine
│       │   └── result_inspector.py        ← Read-only access
│       ├── agents/
│       │   └── research_models.py         ← Unified model interface
│       └── api/
│           └── experiments.py             ← Refactored endpoints
├── experiments/                            ← Artifact storage
│   ├── exp_abc123/
│   │   ├── exp_abc123_config.json
│   │   ├── exp_abc123_metrics.json
│   │   ├── exp_abc123_summary.csv
│   │   └── exp_abc123_logs.csv
│   └── ...
└── notebooks/
    └── analysis.ipynb                    ← Post-hoc analysis
```

---

## Key Principles (PhD-Ready)

1. **All results on disk** → No in-memory state
2. **Reproducible seeds** → Same seed = same results
3. **Complete audit trail** → Every step logged
4. **Offline analysis** → Jupyter, not dashboards
5. **Fair comparison** → All models use same infrastructure
6. **Research-first** → UI is auxiliary only

---

## Next Steps

1. ✅ Artifact schema designed
2. ✅ Runner implemented
3. ✅ Models unified
4. ✅ API refactored
5. ⏳ Angular UI updated to read from disk
6. ⏳ Jupyter analysis notebooks created
7. ⏳ Reproducibility tests added

**Status:** Backend ready for research. UI awaiting integration.
