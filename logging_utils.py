"""
Structured logging for BESS experiments.
Tracks: price, SoC, action, reward, cumulative revenue at each step.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


class EpisodeLogger:
    """Log detailed trajectory for each episode to CSV and JSON."""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_data: List[Dict[str, Any]] = []
        self.cumulative_revenue = 0.0

    def log_step(
        self,
        timestep: int,
        price: float,
        soc: float,
        action: float,
        reward: float,
        info: Dict[str, Any] = None,
    ):
        """Log a single environment step."""
        self.cumulative_revenue += reward
        
        # Classify action discretely for readability
        if action > 0.1:
            action_label = "charge"
        elif action < -0.1:
            action_label = "discharge"
        else:
            action_label = "idle"

        step_record = {
            "timestep": timestep,
            "price_euro_per_mwh": round(price, 2),
            "soc_percent": round(soc * 100, 1),
            "action_kw": round(action, 2),
            "action_label": action_label,
            "reward": round(reward, 2),
            "cumulative_revenue_euro": round(self.cumulative_revenue, 2),
        }
        if info:
            step_record.update(info)

        self.episode_data.append(step_record)

    def save_episode(self, episode_id: int, agent_name: str = "agent"):
        """Save episode trajectory to CSV and JSON."""
        if not self.episode_data:
            return None

        # Sanitize filenames
        filename_base = f"episode_{episode_id:04d}_{agent_name}"
        
        csv_path = self.output_dir / f"{filename_base}.csv"
        json_path = self.output_dir / f"{filename_base}.json"

        # Write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.episode_data[0].keys())
            writer.writeheader()
            writer.writerows(self.episode_data)

        # Write JSON
        with open(json_path, "w") as f:
            json.dump(self.episode_data, f, indent=2)

        result = {
            "csv": str(csv_path),
            "json": str(json_path),
            "total_revenue": self.cumulative_revenue,
            "num_steps": len(self.episode_data),
        }

        # Reset for next episode
        self.episode_data = []
        self.cumulative_revenue = 0.0

        return result


class ExperimentSummary:
    """Aggregate results across all episodes into summary tables."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def add_episode_result(
        self,
        agent_name: str,
        episode: int,
        total_revenue: float,
        num_steps: int,
        extra_metrics: Dict[str, float] = None,
    ):
        """Add one episode's summary to results."""
        if agent_name not in self.results:
            self.results[agent_name] = []

        record = {
            "episode": episode,
            "total_revenue_euro": round(total_revenue, 2),
            "avg_revenue_per_step": round(total_revenue / num_steps, 2) if num_steps > 0 else 0,
            "num_steps": num_steps,
        }
        if extra_metrics:
            record.update(
                {k: round(v, 3) if isinstance(v, float) else v for k, v in extra_metrics.items()}
            )

        self.results[agent_name].append(record)

    def save_summary(self, filename: str = "summary.csv"):
        """Save aggregate results to CSV."""
        summary_path = self.output_dir / filename
        
        # Flatten results for easy comparison
        all_rows = []
        for agent_name, episodes in self.results.items():
            for ep_record in episodes:
                row = {"agent": agent_name, **ep_record}
                all_rows.append(row)

        if all_rows:
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)

        return str(summary_path)

    def print_summary(self):
        """Print summary stats to terminal."""
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)

        for agent_name, episodes in self.results.items():
            revenues = [ep["total_revenue_euro"] for ep in episodes]
            avg_rev = np.mean(revenues) if revenues else 0
            std_rev = np.std(revenues) if len(revenues) > 1 else 0
            
            print(f"\n{agent_name}:")
            print(f"  Episodes: {len(episodes)}")
            print(f"  Avg Revenue: €{avg_rev:.2f}")
            print(f"  Std Dev: €{std_rev:.2f}")
            print(f"  Min: €{min(revenues):.2f}, Max: €{max(revenues):.2f}")

        print("\n" + "="*70)


def print_step_summary(
    timestep: int,
    price: float,
    soc: float,
    action: float,
    cumulative_revenue: float,
):
    """Print one step's state to terminal for live monitoring."""
    action_label = "CHARGE" if action > 0.1 else ("DISCHARGE" if action < -0.1 else "IDLE")
    print(
        f"[t={timestep:4d}] Price: €{price:6.2f}/MWh | "
        f"SoC: {soc*100:5.1f}% | "
        f"Action: {action_label:9s} ({action:+6.2f} kW) | "
        f"Cumulative Revenue: €{cumulative_revenue:8.2f}"
    )
