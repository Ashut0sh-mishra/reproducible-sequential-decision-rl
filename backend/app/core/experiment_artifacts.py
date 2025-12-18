"""
Research-Grade Experiment Artifact Schema & Management

This module defines the standardized format for all experimental results.
Every experiment produces reproducible, auditable artifacts on disk.

RESEARCH-CRITICAL: All primary results live in CSV/JSON files, not in-memory.
UI and dashboards ONLY read from these files.
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class ExperimentConfig:
    """
    Immutable experiment configuration. Saved with every experiment for audit trail.
    
    RESEARCH-CRITICAL: This is the ground truth for reproducibility.
    """
    experiment_id: str
    timestamp: str
    seed: int
    model_name: str
    model_params: Dict[str, Any]
    environment_params: Dict[str, Any]
    num_episodes: int
    num_steps_per_episode: int
    
    def to_json_file(self, output_dir: Path):
        """Save config as JSON for audit trail."""
        config_path = output_dir / f"exp_{self.experiment_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return str(config_path)


class ExperimentArtifactManager:
    """
    Manages the complete lifecycle of experiment artifacts.
    Every experiment produces four files:
    
    1. exp_<id>_config.json      → Reproducibility (seeds, hyperparams)
    2. exp_<id>_metrics.json     → Aggregate episode metrics
    3. exp_<id>_summary.csv      → Summary stats (mean, std, min, max)
    4. exp_<id>_logs.csv         → Step-by-step trajectories
    
    RESEARCH-CRITICAL: These files are the primary research output.
    """
    
    def __init__(self, experiment_dir: Path = Path("experiments")):
        """
        Args:
            experiment_dir: Root directory for all experiment artifacts
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment_dir(self, experiment_id: str) -> Path:
        """Create a subdirectory for this experiment's artifacts."""
        exp_path = self.experiment_dir / experiment_id
        exp_path.mkdir(parents=True, exist_ok=True)
        return exp_path
    
    def save_config(
        self,
        experiment_id: str,
        config: ExperimentConfig,
    ) -> Path:
        """
        Save experiment configuration (ground truth for reproducibility).
        
        RESEARCH-CRITICAL: This enables anyone to reproduce the exact experiment.
        """
        exp_dir = self.create_experiment_dir(experiment_id)
        config_path = exp_dir / f"exp_{experiment_id}_config.json"
        
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        return config_path
    
    def save_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, Any],
    ) -> Path:
        """
        Save per-episode metrics (episode-level aggregation).
        
        Example metrics dict:
        {
            "episodes": [
                {"episode": 0, "total_reward": 1234.56, "final_soc": 0.45, ...},
                {"episode": 1, "total_reward": 1456.78, "final_soc": 0.52, ...},
            ],
            "statistics": {
                "mean_reward": 1345.67,
                "std_reward": 111.11,
                "min_reward": 1234.56,
                "max_reward": 1456.78,
            }
        }
        """
        exp_dir = self.create_experiment_dir(experiment_id)
        metrics_path = exp_dir / f"exp_{experiment_id}_metrics.json"
        
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics_path
    
    def save_summary(
        self,
        experiment_id: str,
        summary: List[Dict[str, Any]],
    ) -> Path:
        """
        Save summary table (one row per episode).
        
        RESEARCH-CRITICAL: Easy to load into pandas for analysis.
        
        Example rows:
        [
            {"episode": 0, "total_reward": 1234.56, "steps": 720, "model": "rule_based"},
            {"episode": 1, "total_reward": 1456.78, "steps": 720, "model": "rule_based"},
        ]
        """
        exp_dir = self.create_experiment_dir(experiment_id)
        summary_path = exp_dir / f"exp_{experiment_id}_summary.csv"
        
        if summary:
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=summary[0].keys())
                writer.writeheader()
                writer.writerows(summary)
        
        return summary_path
    
    def save_logs(
        self,
        experiment_id: str,
        logs: List[Dict[str, Any]],
    ) -> Path:
        """
        Save step-by-step trajectory logs (complete audit trail).
        
        RESEARCH-CRITICAL: Enables post-hoc analysis and policy inspection.
        
        Example rows:
        [
            {
                "episode": 0, "step": 0, "price": 67.5, "soc": 0.50, 
                "action": 5.0, "reward": 12.34, "cumulative_reward": 12.34
            },
            {...}
        ]
        """
        exp_dir = self.create_experiment_dir(experiment_id)
        logs_path = exp_dir / f"exp_{experiment_id}_logs.csv"
        
        if logs:
            with open(logs_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                writer.writeheader()
                writer.writerows(logs)
        
        return logs_path
    
    def save_all_artifacts(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        metrics: Dict[str, Any],
        summary: List[Dict[str, Any]],
        logs: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Save all artifacts for one complete experiment.
        
        Returns:
            Dict mapping artifact type to file path
        """
        return {
            "config": str(self.save_config(experiment_id, config)),
            "metrics": str(self.save_metrics(experiment_id, metrics)),
            "summary": str(self.save_summary(experiment_id, summary)),
            "logs": str(self.save_logs(experiment_id, logs)),
        }
    
    def load_config(self, experiment_id: str) -> ExperimentConfig:
        """Load experiment config for reproducibility verification."""
        config_path = self.experiment_dir / experiment_id / f"exp_{experiment_id}_config.json"
        with open(config_path) as f:
            data = json.load(f)
        return ExperimentConfig(**data)
    
    def load_summary(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Load summary table for analysis."""
        summary_path = self.experiment_dir / experiment_id / f"exp_{experiment_id}_summary.csv"
        rows = []
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows
    
    def load_logs(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Load step-by-step logs for detailed analysis."""
        logs_path = self.experiment_dir / experiment_id / f"exp_{experiment_id}_logs.csv"
        rows = []
        with open(logs_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows
    
    def list_experiments(self) -> List[str]:
        """List all experiment IDs."""
        if not self.experiment_dir.exists():
            return []
        return [d.name for d in self.experiment_dir.iterdir() if d.is_dir()]


if __name__ == "__main__":
    print("=" * 80)
    print("Experiment Artifact Schema")
    print("=" * 80)
    print()
    print("Every experiment produces:")
    print("  1. exp_<id>_config.json    - Reproducibility audit trail")
    print("  2. exp_<id>_metrics.json   - Aggregate statistics")
    print("  3. exp_<id>_summary.csv    - Episode-level results")
    print("  4. exp_<id>_logs.csv       - Step-by-step trajectories")
    print()
    print("These files are PRIMARY RESEARCH OUTPUT.")
    print("UI/dashboards read ONLY from these files.")
    print("No results exist in-memory or in UI state.")
    print()
