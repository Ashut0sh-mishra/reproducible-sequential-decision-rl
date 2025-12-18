Deterministic Reinforcement Learning for Sequential Control: A Reproducible Research Framework

## 1. Research Motivation and Scope

This repository provides a research-oriented framework for controlled experimental evaluation of algorithms for sequential decision-making. The design prioritises reproducibility, methodological clarity and auditable evidence suitable for PhD-level review.

**Research question**

To what extent do reinforcement learning policies improve upon heuristic and statistical baselines for sequential control problems under identical, reproducible experimental conditions?

**Scope**
- General sequential-control research: a synthetic control environment is used as an interpretable benchmark.
- Comparison of heuristic, statistical and deep RL policies.
- Deterministic experiments with fixed random seeds and complete audit trails.

**What this is not**
- Not a real-world energy optimisation study.
- Not a deployment system.
- Not a UI-driven evaluation (UI is auxiliary and read-only).

## 2. System architecture

### Research pipeline

```
┌─────────────────────────────────────────────────┐
│ Models                                          │
│ • Rule-based (heuristic)                         │
│ • Random baseline                                │
│ • Logistic regression                            │
│ • Random forest                                  │
│ • DQN/PPO (deep RL, trained offline)             │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Deterministic experiment runner                  │
│ • Fixed seeds (numpy, random, torch, CUDA)      │
│ • Episodic evaluation (preserved count per run) │
│ • Complete step-level logging                    │
│ • All randomness documented in config           │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ PRIMARY RESEARCH ARTIFACTS (on disk)            │
│ • exp_<id>_config.json     [reproducibility]    │
│ • exp_<id>_metrics.json    [aggregate stats]    │
│ • exp_<id>_summary.csv     [episode results]    │
│ • exp_<id>_logs.csv        [step trajectories]  │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Offline analysis (Jupyter notebooks)            │
│ • notebooks/01_experiment_analysis.ipynb        │
│ • notebooks/02_policy_behavior_analysis.ipynb   │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│ Optional UI (read-only)                         │
│ • File-based inspection only                     │
│ • Does not contribute to experimental results   │
└─────────────────────────────────────────────────┘
```

### Component responsibilities

The repository contains both core components (those that directly generate research evidence) and auxiliary components (inspection, API). Core components must not be modified by reviewers attempting to reproduce results.

- `app/core/research_runner.py` — Experiment orchestration and seeding (core)
- `app/agents/research_models.py` — Model implementations; unified interface (core)
- `app/core/experiment_artifacts.py` — Artifact schema and serialization (core)
- `envs/bess_env.py` — Synthetic, interpretable benchmark environment (core)
- `scripts/run_batch_experiments.py` — Batch orchestration for reproducible runs (core)
- `notebooks/` — Offline analysis and evidence generation (core)
- `backend/app/api/` — REST endpoints (auxiliary)
- Angular frontend — Inspection only (auxiliary)

## 3. Deterministic experiment design

### Reproducibility guarantees

All experiments are executed under controlled randomness. The runner sets seeds for all relevant sources and persists the seed and all parameters in `exp_<id>_config.json`.

- `np.random.seed(seed)`
- `random.seed(seed)`
- `torch.manual_seed(seed)` (+ CUDA where applicable)

### Experimental setup (kept unchanged)

The implementation uses conservative, explicitly documented choices; these are preserved exactly in the code and in saved configurations.

- Episodes per run: 500
- Horizon per episode: 720 steps
- Seeds used (example): 42, 123, 999

The environment is a synthetic, interpretable benchmark that resembles a single-device sequential control task; it is not presented as a validated domain model for operational deployment.

## 4. Primary research artifacts

Every experiment produces the following artifacts in `results/<model>/<experiment_id>/` and these files are the primary evidence for any claim:

- `exp_<id>_config.json` — Immutable configuration (seed, parameters, environment settings)
- `exp_<id>_metrics.json` — Aggregate statistics (means, stds, additional metrics)
- `exp_<id>_summary.csv` — Episode-level rows (one row per episode)
- `exp_<id>_logs.csv` — Step-level trajectories (audit trail)

Reviewers should rely on these files for verification. Notebooks and auxiliary interfaces read from these artifacts only.

## 5. Offline analysis notebooks

Two notebooks accompany the repository and perform all analysis from saved artifacts:

- `notebooks/01_experiment_analysis.ipynb` — Aggregation across seeds, tables of metrics, and static plots for comparison (classification metrics for statistical baselines; reward statistics and success rate for policies).
- `notebooks/02_policy_behavior_analysis.ipynb` — Step-level analysis of action distributions and exploratory classification of behaviour around adverse events.

These notebooks are descriptive and intended for reproducible evidence generation; they do not re-run experiments or change stored artifacts.

## 6. Quick start (reproducible minimal steps)

Install dependencies (from project root):

```powershell
pip install -r requirements.txt
```

Run the batch script (this uses the existing runner and preserves the experimental design):

```powershell
python scripts/run_batch_experiments.py
```

Open the analysis notebooks for offline inspection:

```powershell
jupyter notebook notebooks/01_experiment_analysis.ipynb
jupyter notebook notebooks/02_policy_behavior_analysis.ipynb
```

## 7. Verification procedure (for reviewers)

Reproducibility checks should follow the artifact-first workflow: run experiments with the same documented seed, locate the `exp_<id>_config.json` and `exp_<id>_metrics.json` files, and compare numerical summaries. The repository includes a `REPRODUCIBILITY_GUIDE.md` with a checklist.

## 8. File structure (high level)

```
battery_rl_project/
├── backend/
│   └── app/
├── envs/                    # synthetic benchmark environment
├── scripts/                 # runner scripts (batch orchestration)
├── notebooks/               # offline analysis
├── results/                 # PRIMARY research artifacts (CSV/JSON)
└── documentation/
```

## 9. Limitations and scope (explicit)

- The synthetic environment is intended as an interpretable benchmark for sequential control research; it is not a validated operational model.
- The repository focuses on reproducibility and fair comparison rather than production deployment.
- No hyperparameter optimisation is reported in the baseline experiments; default, documented settings are used.

## 10. Tone and usage

This document is written for an academic audience: supervisors, reviewers and researchers. It emphasises conservative interpretation of results, reproducibility, and a clear separation between core experimental code and auxiliary inspection tools.

For detailed verification steps, see `REPRODUCIBILITY_GUIDE.md` and the notebooks in `notebooks/`.


