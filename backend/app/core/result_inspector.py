"""
Result Inspection Module (Read-Only)

RESEARCH-CRITICAL: This is a read-only interface to saved artifacts.
No computation happens here—all results pre-computed and on disk.
UI uses this to display static data only.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional


class ExperimentResultsInspector:
    """
    Read-only interface for accessing saved experiment results.
    
    All methods are pure (no side effects).
    All data comes from disk artifacts.
    """
    
    def __init__(self, experiments_dir: Path = Path("experiments")):
        self.experiments_dir = Path(experiments_dir)
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get high-level summary of an experiment.
        """
        exp_dir = self.experiments_dir / experiment_id
        
        config_path = exp_dir / f"exp_{experiment_id}_config.json"
        metrics_path = exp_dir / f"exp_{experiment_id}_metrics.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        
        with open(config_path) as f:
            config = json.load(f)
        
        with open(metrics_path) as f:
            metrics = json.load(f)
        
        return {
            "experiment_id": experiment_id,
            "timestamp": config["timestamp"],
            "model": config["model_name"],
            "seed": config["seed"],
            "num_episodes": config["num_episodes"],
            "statistics": metrics.get("statistics", {}),
        }
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple experiments side-by-side.
        
        RESEARCH-CRITICAL: Enables fair model comparison.
        """
        rows = []
        for exp_id in experiment_ids:
            try:
                summary = self.get_experiment_summary(exp_id)
                stats = summary.pop("statistics", {})
                row = {**summary, **stats}
                rows.append(row)
            except FileNotFoundError:
                continue
        
        return pd.DataFrame(rows)
    
    def load_summary_table(self, experiment_id: str) -> pd.DataFrame:
        """
        Load episode-level summary (one row per episode).
        """
        exp_dir = self.experiments_dir / experiment_id
        summary_path = exp_dir / f"exp_{experiment_id}_summary.csv"
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary for {experiment_id} not found")
        
        return pd.read_csv(summary_path)
    
    def load_logs_table(self, experiment_id: str) -> pd.DataFrame:
        """
        Load step-by-step logs.
        
        RESEARCH-CRITICAL: Complete audit trail.
        """
        exp_dir = self.experiments_dir / experiment_id
        logs_path = exp_dir / f"exp_{experiment_id}_logs.csv"
        
        if not logs_path.exists():
            raise FileNotFoundError(f"Logs for {experiment_id} not found")
        
        return pd.read_csv(logs_path)
    
    def get_artifact_paths(self, experiment_id: str) -> Dict[str, str]:
        """
        Get paths to all artifacts (for file-based access).
        """
        exp_dir = self.experiments_dir / experiment_id
        
        return {
            "config": str(exp_dir / f"exp_{experiment_id}_config.json"),
            "metrics": str(exp_dir / f"exp_{experiment_id}_metrics.json"),
            "summary": str(exp_dir / f"exp_{experiment_id}_summary.csv"),
            "logs": str(exp_dir / f"exp_{experiment_id}_logs.csv"),
        }
    
    def list_all_experiments(self) -> List[Dict[str, Any]]:
        """
        List all available experiments with summaries.
        """
        if not self.experiments_dir.exists():
            return []
        
        experiments = []
        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if exp_dir.is_dir():
                try:
                    summary = self.get_experiment_summary(exp_dir.name)
                    experiments.append(summary)
                except:
                    pass
        
        return experiments


if __name__ == "__main__":
    print("=" * 80)
    print("Result Inspection Module (Read-Only)")
    print("=" * 80)
    print()
    print("All methods read from saved artifacts only.")
    print("No computation, no side effects.")
    print("Pure data access for UI display.")
    print()
