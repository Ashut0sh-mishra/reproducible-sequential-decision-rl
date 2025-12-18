"""
Batch experiment runner for research-grade reproducible experiments.

This script:
- Iterates over models × seeds
- Uses the existing `ResearchExperimentRunner` (no architectural changes)
- Trains logistic regression and random forest baselines on the same dataset
- Wraps scikit-learn classifiers with a `ResearchModel` adapter
- Runs `num_episodes` episodes per run (deterministic via seeds)
- Ensures all artifacts are saved under `results/<model_name>/exp_<id>/`
- Computes classification metrics (Accuracy, Precision, Recall, F1) for baselines
- Computes reward statistics and success rate per run

NOTES (research-critical):
- This file does not change core architecture. It uses `ResearchExperimentRunner`'s
  public API and `create_research_experiment_config` helper.
- The dataset and train/test split are shared across models (fixed split seed)
- Per-run randomness (model initialization, environment, training) is controlled
  by the experiment `seed` (one of the seeds list below), ensuring reproducibility.

Run (from project root):
    python scripts/run_batch_experiments.py

"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root and backend are importable
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT))

import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.core.research_runner import (
    ResearchExperimentRunner,
    create_research_experiment_config,
)
from app.core.experiment_artifacts import ExperimentConfig, ExperimentArtifactManager
from app.agents.research_models import (
    RuleBasedModel,
    RandomModel,
    GreedyModel,
    DummyRLModel,
)
from envs.bess_env import BessEnv
from data.generate_prices import generate_price_series

# ------------------------------ Experiment Plan ------------------------------
MODELS = [
    "rule_based",
    "logistic_regression",
    "random_forest",
    "dqn",  # placeholder RL (DummyRLModel) to keep experiment pipeline complete
]

SEEDS = [42, 123, 999]
NUM_EPISODES = 500  # fixed number of episodes per run (research decision)
NUM_STEPS_PER_EPISODE = 720  # typical monthly horizon (24*30)
RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(exist_ok=True)

# Dataset parameters (shared across models)
PRICE_DAYS = 30  # produce 30 days of hourly prices -> 720 steps
DATASET_SEED_FOR_SPLIT = 0  # fixed split seed: ensures same train/test across models
BASE_PRICE = 50

# -----------------------------------------------------------------------------
# Utility: Adapter to wrap scikit-learn classifiers as ResearchModel-compatible
# -----------------------------------------------------------------------------
class SklearnClassifierWrapper:
    """Adapter to use sklearn classifier as a ResearchModel-like object.

    The wrapper exposes .act(observation) -> float where observation is the
    environment observation: [soc, price_now, price_next, hour]. It maps a
    binary prediction to discrete actions: class 1 -> discharge (-p_max),
    class 0 -> charge (+p_max).
    """

    def __init__(self, clf, p_max: float = 10.0):
        self.clf = clf
        self.p_max = float(p_max)

    def reset(self):
        # stateless classifier
        return None

    def act(self, observation: np.ndarray) -> float:
        # Features must be consistent with training: [price_now, price_next, hour]
        price_now = float(observation[1])
        price_next = float(observation[2])
        hour = float(observation[3])
        x = np.array([[price_now, price_next, hour]])
        pred = int(self.clf.predict(x)[0])
        # Map to continuous action: 0 -> charge, 1 -> discharge
        return -self.p_max if pred == 1 else self.p_max

# -----------------------------------------------------------------------------
# Helper: compute classification metrics (binary)
# -----------------------------------------------------------------------------

def compute_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "precision": float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
        "f1": float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
    }

# -----------------------------------------------------------------------------
# Prepare shared dataset (features and binary label)
# Feature design (explicit & simple for reproducibility):
#  - X: [price_now, price_next, hour]
#  - y: 1 if price_next > price_now (suggesting discharge), else 0
# -----------------------------------------------------------------------------
print("Preparing shared dataset...")
price_df = generate_price_series(days=PRICE_DAYS, base_price=BASE_PRICE, seed=0)
prices = price_df["price"].values  # numpy array
hours = np.arange(len(prices)) % 24

X = []
Y = []
for t in range(len(prices) - 1):
    price_t = float(prices[t])
    price_next = float(prices[t + 1])
    hour = float(hours[t])
    X.append([price_t, price_next, hour])
    Y.append(1 if price_next > price_t else 0)

X = np.array(X)
Y = np.array(Y)

# Fixed train/test split used for classifier evaluation and training.
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=DATASET_SEED_FOR_SPLIT, shuffle=True
)

print(f"Dataset prepared: {len(X_train)} train / {len(X_test)} test rows")

# -----------------------------------------------------------------------------
# Main loop: iterate models x seeds
# -----------------------------------------------------------------------------

for model_name in MODELS:
    model_dir = RESULTS_ROOT / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        # Minimal console output required by user
        print(f"Running model={model_name} seed={seed} -> outputs under {model_dir}")

        # Reproducibility: fix numpy/random/torch seeds used here (training & env)
        np.random.seed(seed)
        random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
        except Exception:
            pass

        # Environment config: same for all models
        environment_params = {
            "price_series": prices.tolist(),
            "battery_capacity_kwh": 50.0,
            "p_max_kw": 10.0,
            "eff": 0.95,
            "cost_per_full_cycle": 10.0,
            "soc_init": 0.5,
        }

        # Model params (kept minimal, no hyperparameter tuning)
        model_params: Dict[str, Any] = {"p_max": 10.0}

        # Build a model_factory for this run. For scikit models we train here,
        # then provide a factory that returns the trained wrapper instance.
        trained_clf = None

        if model_name == "logistic_regression":
            clf = LogisticRegression(solver="liblinear", random_state=seed)
            clf.fit(X_train, y_train)
            trained_clf = clf
            model_factory = lambda name, params: SklearnClassifierWrapper(trained_clf, p_max=params.get("p_max", 10.0))
            # For metrics later, get preds on test set
            y_pred_test = trained_clf.predict(X_test)
            classification_metrics = compute_classification_metrics(y_test, y_pred_test)

        elif model_name == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, random_state=seed)
            clf.fit(X_train, y_train)
            trained_clf = clf
            model_factory = lambda name, params: SklearnClassifierWrapper(trained_clf, p_max=params.get("p_max", 10.0))
            y_pred_test = trained_clf.predict(X_test)
            classification_metrics = compute_classification_metrics(y_test, y_pred_test)

        elif model_name == "rule_based":
            # Use existing RuleBasedModel implementation
            model_factory = lambda name, params: RuleBasedModel(params)
            # We can compute classification metrics by applying rule-based logic to dataset
            rb = RuleBasedModel({"low_threshold": 50.0, "high_threshold": 100.0, "p_max": 10.0})
            preds = []
            for x_row in X_test:
                # Build observation: [soc, price_now, price_next, hour]
                obs = np.array([0.5, x_row[0], x_row[1], x_row[2]])
                act = rb.act(obs)
                pred_label = 1 if act < 0 else 0
                preds.append(pred_label)
            classification_metrics = compute_classification_metrics(y_test, np.array(preds))

        elif model_name == "dqn":
            # Placeholder DummyRLModel (no training). Keep deterministic by seed where possible.
            model_factory = lambda name, params: DummyRLModel(params)
            classification_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        else:
            raise ValueError(f"Unknown model in plan: {model_name}")

        # Instantiate runner with per-model artifact directory so outputs are grouped
        artifact_dir_for_model = str(model_dir)
        runner = ResearchExperimentRunner(env_factory=BessEnv, model_factory=model_factory, artifact_dir=artifact_dir_for_model)

        # Create reproducible config
        config = create_research_experiment_config(
            model_name=model_name,
            model_params=model_params,
            environment_params=environment_params,
            seed=seed,
            num_episodes=NUM_EPISODES,
            num_steps_per_episode=NUM_STEPS_PER_EPISODE,
        )

        # Run the experiment (synchronous, artifacts written by runner)
        result = runner.run_experiment(
            config=config,
            model_name=model_name,
            model_params=model_params,
            environment_params=environment_params,
            num_episodes=NUM_EPISODES,
        )

        # Post-process metrics: augment with classification metrics & success rate
        artifact_paths = result["artifacts"]
        metrics_path = Path(artifact_paths["metrics"]) if "metrics" in artifact_paths else (Path(artifact_dir_for_model) / config.experiment_id / f"exp_{config.experiment_id}_metrics.json")

        # Load existing metrics
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        # Add classification metrics (for baselines and adapted models); keep consistent keys
        metrics_data["classification_metrics"] = classification_metrics

        # Compute success rate: fraction of episodes with total_reward > 0
        ep_rewards = [ep["total_reward"] for ep in metrics_data.get("episodes", [])]
        success_rate = float(round(sum(1 for r in ep_rewards if r > 0) / len(ep_rewards), 4)) if ep_rewards else 0.0
        metrics_data["success_rate"] = success_rate

        # Overwrite metrics file with augmented metrics (research artifact)
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        # Minimal console output (again) with artifact locations
        print(f"Completed model={model_name} seed={seed} -> experiment_id={config.experiment_id}")
        print(f"Artifacts: \n  config: {artifact_paths.get('config')}\n  metrics: {metrics_path}\n  summary: {artifact_paths.get('summary')}\n  logs: {artifact_paths.get('logs')}\n")

print("Batch experiments finished.")
