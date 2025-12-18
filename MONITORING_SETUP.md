# Monitoring & Logging System – Research Ready

## Summary of Changes

You now have a **PhD-ready structured logging system** instead of a web dashboard. Here's what was added:

---

## New Files

| File | Purpose |
|------|---------|
| **logging_utils.py** | Core logging: `EpisodeLogger`, `ExperimentSummary`, terminal print helpers |
| **experiment_runner.py** | Orchestrates training, logs trajectories, runs multi-agent experiments |
| **notebooks/analysis_template.ipynb** | Post-hoc Jupyter notebook for inspecting logs and comparing agents |
| **LOGGING_GUIDE.md** | Full workflow documentation (this guide) |

---

## How It Works

### During Experiments (Terminal Output)
```
[t=  0] Price: € 42.15/MWh | SoC:  50.0% | Action: CHARGE   (+5.00 kW) | Cumulative Revenue: €    0.00
[t= 24] Price: € 89.32/MWh | SoC:  75.2% | Action: DISCHARGE (-5.00 kW) | Cumulative Revenue: €   156.42
[EPISODE 0 COMPLETE] Agent: baseline_rule
  Total Revenue: €456.78
  Steps: 720
  Logs: logs/episode_0000_baseline_rule.csv
```

### Saved Artifacts

After each experiment, you get:

1. **Step-by-step logs (CSV/JSON)**
   ```
   logs/episode_0000_baseline_rule.csv
   logs/episode_0001_baseline_rule.csv
   logs/episode_0002_ppo_agent.csv
   ...
   ```
   Each row: timestep, price, SoC, action, reward, cumulative_revenue

2. **Aggregate summary (CSV)**
   ```
   reports/summary.csv
   
   agent,episode,total_revenue_euro,avg_revenue_per_step,num_steps,final_soc
   baseline_rule,0,456.78,0.63,720,0.52
   baseline_rule,1,523.45,0.73,720,0.48
   ppo_agent,0,678.90,0.94,720,0.55
   ```

3. **Terminal summary**
   ```
   ======================================================================
   EXPERIMENT SUMMARY
   ======================================================================
   
   baseline_rule:
     Episodes: 2
     Avg Revenue: €490.12
     Std Dev: €47.33
     Min: €456.78, Max: €523.45
   
   ppo_agent:
     Episodes: 1
     Avg Revenue: €678.90
     Std Dev: €0.00
     Min: €678.90, Max: €678.90
   ```

---

## Post-Hoc Analysis (Jupyter)

Open **notebooks/analysis_template.ipynb** to:

1. **Load & List Logs**
   ```python
   csv_logs = sorted(LOG_DIR.glob("*.csv"))
   # Output: ['episode_0000_baseline_rule.csv', ...]
   ```

2. **Inspect Single Episode**
   ```python
   episode_df = pd.read_csv(csv_logs[0])
   print(episode_df.head())  # Tabular step-by-step data
   ```

3. **Plot Trajectory** (3-panel figure)
   - Panel 1: Electricity price over time
   - Panel 2: State of charge with action coloring (charge=green, discharge=red)
   - Panel 3: Cumulative revenue trajectory

4. **Compare Agents**
   ```
   Agent Performance Comparison:
           mean      std      min      max  episodes
   agent                                          
   baseline_rule  490.12   47.33  456.78  523.45        2
   ppo_agent      678.90    0.00  678.78  678.90        1
   ```

---

## Quick Start Example

### 1. Run an Experiment
```python
from experiment_runner import SimpleExperimentRunner, SimpleBaselineController
from envs.bess_env import BessEnv
import numpy as np

np.random.seed(42)

# Create environment with price data
prices = np.random.uniform(20, 150, 720)  # 30 days
env = BessEnv(price_series=prices)

# Create agents
baseline = SimpleBaselineController(low_threshold=50, high_threshold=100)
agents = {"baseline_rule": baseline}

# Run
runner = SimpleExperimentRunner(env)
runner.run_experiment(agents, num_episodes=2, verbose=True)
```

**Output:**
```
logs/episode_0000_baseline_rule.csv
logs/episode_0001_baseline_rule.csv
reports/summary.csv
```

### 2. Analyze in Jupyter
Open **analysis_template.ipynb**, run all cells. You'll see:
- Episode tables
- Trajectory plots
- Summary statistics
- Agent comparisons

---

## Key Design Choices

✅ **No live dashboard** – Focuses on data logging, not real-time UI  
✅ **Reproducible** – All experiments seeded, outputs deterministic  
✅ **Research-ready** – CSV/JSON for easy external analysis  
✅ **Extensible** – Easy to add new agents, metrics, or analysis  
✅ **Plain language** – Terminal output is human-readable  

---

## What's Next?

1. **Integrate your RL agent**: Wrap training output in an `act(obs)` interface
2. **Add more baselines**: MPC optimizer, random policy, etc.
3. **Run full experiments**: Compare agents across multiple price scenarios
4. **Export results**: Use Jupyter notebook to generate tables/figures for your PhD thesis

---

## FAQ

**Q: Where's the live dashboard?**  
A: Deliberately omitted. PhD work prioritizes reproducibility over real-time UI. Use terminal `verbose=True` or check Jupyter after experiments complete.

**Q: How do I extend this?**  
A: Subclass `EpisodeLogger` or `ExperimentSummary` for custom metrics. See code comments.

**Q: Can I use real price data?**  
A: Yes—load CSV into `BessEnv(price_series=...)` instead of synthetic data.

**Q: How do I track hyperparameter sweeps?**  
A: Save config alongside logs (JSON metadata). Modify `ExperimentSummary` to include config columns.

---

**Status**: ✅ Logging infrastructure ready. Ready to integrate RL agent training.
