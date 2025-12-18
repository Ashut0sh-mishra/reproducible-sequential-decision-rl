# Reproducibility & Audit Trail Guide

## Why This Matters (PhD Context)

Reviewers will ask: "How do I know your results are real?"

**Answer:** Run the same config → get identical results.

This document explains how the system ensures reproducibility and provides audit trails.

---

## Reproducibility Layers

### Layer 1: Deterministic Seeding

**Every random number source is controlled:**

```python
def set_seeds(seed: int):
    """Set ALL random sources to the same seed."""
    np.random.seed(seed)           # NumPy
    random.seed(seed)              # Python random
    torch.manual_seed(seed)        # PyTorch
    torch.cuda.manual_seed(seed)   # CUDA if available
```

**Called once at start of every experiment:**
```python
config = create_research_experiment_config(seed=42)
ResearchExperimentRunner.set_seeds(config.seed)
# Now: All randomness is deterministic
```

**Result:** Same seed → bit-exact identical results (in CPU mode).

---

### Layer 2: Configuration Audit Trail

**Every experiment config saved to JSON:**

```json
{
  "experiment_id": "a1b2c3d4",
  "timestamp": "2025-12-17T10:30:00",
  "seed": 42,
  "model_name": "rule_based",
  "model_params": {
    "low_threshold": 50.0,
    "high_threshold": 100.0,
    "p_max": 10.0
  },
  "environment_params": {
    "battery_capacity_kwh": 50.0,
    "price_series": "synthetic_2025_12_17.csv"
  },
  "num_episodes": 3,
  "num_steps_per_episode": 720
}
```

**Why it matters:**
- Reviewer can load this JSON
- Reviewer can recreate exact experiment
- No ambiguity about hyperparameters

---

### Layer 3: Step-by-Step Logs

**Every environment step logged to CSV:**

```csv
episode,step,obs_0,obs_1,obs_2,obs_3,action,reward,cumulative_reward,price,soc
0,0,0.50,67.50,72.30,0.0,5.00,12.34,12.34,67.50,0.50
0,1,0.55,72.30,71.20,-1.0,-5.00,8.90,21.24,72.30,0.55
0,2,0.51,71.20,70.15,-2.0,0.00,-2.10,19.14,71.20,0.51
```

**Why it matters:**
- Reviewer can trace every decision
- Catch bugs or unexpected behavior
- Verify reward computation

---

### Layer 4: Aggregate Metrics

**Summary statistics saved to JSON:**

```json
{
  "episodes": [
    {"episode": 0, "total_reward": 1234.56, "num_steps": 720},
    {"episode": 1, "total_reward": 1456.78, "num_steps": 720},
    {"episode": 2, "total_reward": 1345.67, "num_steps": 720}
  ],
  "statistics": {
    "mean_reward": 1345.67,
    "std_reward": 111.11,
    "min_reward": 1234.56,
    "max_reward": 1456.78,
    "median_reward": 1345.67
  }
}
```

**Why it matters:**
- Easy to extract for thesis figures
- Standard format (loadable by any tool)
- Enables statistical tests

---

## Verification Checklist

### Before Publishing Results

- [ ] Experiment config saved? (`exp_<id>_config.json`)
- [ ] Seed documented in config?
- [ ] Step-by-step logs saved? (`exp_<id>_logs.csv`)
- [ ] Reproducibility verified?
  ```bash
  # Run same config twice, compare results
  result1 = runner.run_experiment(config)
  result2 = runner.run_experiment(config)
  assert result1["summary_stats"] == result2["summary_stats"]
  ```
- [ ] No hard-coded random values?
- [ ] All hyperparameters in config (not code)?

---

## Common Pitfalls

### ❌ Bad: Seed set in model class
```python
class MyModel:
    def __init__(self):
        np.random.seed(42)  # Hard-coded! Not reproducible!
```

### ✅ Good: Seed set before experiment
```python
config = create_research_experiment_config(seed=42, ...)
ResearchExperimentRunner.set_seeds(config.seed)
runner.run_experiment(config)
```

---

### ❌ Bad: Hyperparams hard-coded
```python
model = RuleBasedModel()  # Defaults used, not documented
```

### ✅ Good: Hyperparams in config
```python
config = create_research_experiment_config(
    model_params={"low_threshold": 50, "high_threshold": 100}
)
model = model_factory(config.model_name, config.model_params)
```

---

### ❌ Bad: No logs saved
```python
result = runner.run_episode(env, model)
print(f"Reward: {result}")  # Result only in terminal!
```

### ✅ Good: All logs saved
```python
result = runner.run_experiment(config)
# Saved: config.json, metrics.json, summary.csv, logs.csv
```

---

## Reproducibility Testing

### Test 1: Identical Results with Same Seed

```python
from app.core.research_runner import ResearchExperimentRunner, create_research_experiment_config

config1 = create_research_experiment_config(
    model_name="rule_based",
    seed=42,
    num_episodes=3,
)

config2 = create_research_experiment_config(
    model_name="rule_based",
    seed=42,
    num_episodes=3,
)

runner = ResearchExperimentRunner(env_factory, model_factory)

result1 = runner.run_experiment(config1)
result2 = runner.run_experiment(config2)

# Verify identical
assert result1["summary_stats"]["mean_reward"] == result2["summary_stats"]["mean_reward"]
print("✓ Reproducibility Test 1: PASSED")
```

### Test 2: Different Results with Different Seeds

```python
config1 = create_research_experiment_config(seed=42, ...)
config2 = create_research_experiment_config(seed=123, ...)

result1 = runner.run_experiment(config1)
result2 = runner.run_experiment(config2)

# Should be different
assert result1["summary_stats"]["mean_reward"] != result2["summary_stats"]["mean_reward"]
print("✓ Reproducibility Test 2: PASSED")
```

### Test 3: Load and Verify Config

```python
from app.core.experiment_artifacts import ExperimentArtifactManager

manager = ExperimentArtifactManager("experiments")
config = manager.load_config("exp_abc123")

print(f"Seed: {config.seed}")
print(f"Model: {config.model_name}")
print(f"Params: {config.model_params}")
# Reviewer can verify everything is documented
```

---

## For PhD Reviewers

**How to verify results are reproducible:**

1. **Get the config:**
   ```bash
   cat experiments/exp_abc123/exp_abc123_config.json
   ```

2. **Check the seed:**
   ```json
   {
     "seed": 42,
     "model_name": "rule_based",
     ...
   }
   ```

3. **Rerun the experiment:**
   ```python
   runner = ResearchExperimentRunner(...)
   result = runner.run_experiment(config)
   ```

4. **Compare results:**
   ```python
   original = json.load(open("experiments/exp_abc123/exp_abc123_metrics.json"))
   verify_exact_match(result, original)
   ```

5. **Inspect logs:**
   ```python
   logs_df = pd.read_csv("experiments/exp_abc123/exp_abc123_logs.csv")
   print(logs_df.head(10))  # See exact actions and rewards
   ```

**Result:** If all checks pass, results are independently reproducible.

---

## Artifact Retention Policy

**Keep all artifacts:**
- ✅ Configs (lightweight JSON)
- ✅ Summary CSVs (for comparison)
- ✅ Logs (for inspection, can be gzipped to save space)
- ✅ Metrics JSON (statistics)

**Example storage:**
```
experiments/
├── exp_001/  (seed 42, rule_based, 2025-12-17)
├── exp_002/  (seed 123, rule_based, 2025-12-17)
├── exp_003/  (seed 42, ppo, 2025-12-18)
└── ...
```

**Archival:**
- For thesis submission, zip all `experiments/` folder
- Include README listing all experiment IDs and configs
- Reviewer can unzip and rerun any experiment

---

## Common Questions

**Q: Do I need to run experiments multiple times?**
A: Yes. Report mean ± std across 3–5 runs with different seeds.

**Q: What if results aren't reproducible?**
A: Debug immediately:
  1. Check seed set correctly
  2. Verify no randomness in model
  3. Check environment initialization
  4. Look for system differences (GPU, OS)

**Q: How do I document this in my thesis?**
A: Say:
> "All experiments are reproducible. A random seed of 42 was used for the control condition (see config in Appendix). Results reported are mean ± std across 3 independent runs (seeds: 42, 123, 456). Complete logs and configs are in the `experiments/` directory."

**Q: What if a reviewer asks for results with a different seed?**
A: Simply rerun:
  ```python
  config.seed = 999
  result = runner.run_experiment(config)
  ```

---

## Summary

| Aspect | Mechanism | Evidence |
|--------|-----------|----------|
| **Reproducibility** | Fixed seeds (numpy, random, torch) | `config.json` seed field |
| **Auditability** | Every step logged | `logs.csv` with all steps |
| **Documentation** | Config saved as JSON | `config.json` with all params |
| **Verification** | Same seed → same results | Re-run test in Jupyter |
| **Comparison** | All models use same infrastructure | `metrics.json` comparable |

**Bottom line for PhD reviewers:** Run the config, get identical results. Reproducibility verified.
