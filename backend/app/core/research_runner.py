"""
Research-Grade Experiment Runner

RESEARCH-CRITICAL: This is the core execution layer for all experiments.
Ensures reproducibility, deterministic behavior, and audit trails.

Every experiment:
1. Sets random seeds (numpy, random, torch)
2. Runs model(s) with consistent state
3. Logs every step
4. Saves all artifacts to disk
5. Produces reproducible results

No UI state. No in-memory results. Pure research pipeline.
"""

import numpy as np
import random
import torch
from pathlib import Path
from typing import Dict, Any, Callable, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid

from app.core.experiment_artifacts import ExperimentConfig, ExperimentArtifactManager


class ResearchExperimentRunner:
    """
    Unified experiment runner for all models and environments.
    
    Responsibilities:
    - Set reproducible seeds
    - Execute episode(s)
    - Log step-by-step trajectories
    - Aggregate metrics
    - Save all artifacts
    """
    
    def __init__(
        self,
        env_factory: Callable,
        model_factory: Callable,
        artifact_dir: str = "experiments",
    ):
        """
        Args:
            env_factory: Callable that returns environment instance
            model_factory: Callable that returns model/agent instance
            artifact_dir: Root directory for experiment artifacts
        """
        self.env_factory = env_factory
        self.model_factory = model_factory
        self.artifact_manager = ExperimentArtifactManager(Path(artifact_dir))
    
    @staticmethod
    def set_seeds(seed: int):
        """
        Set ALL random seeds for reproducibility.
        
        RESEARCH-CRITICAL: Call at start of every experiment.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def run_episode(
        self,
        env,
        model,
        episode_id: int,
        max_steps: int = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Run one episode and collect step-by-step logs.
        
        Args:
            env: Environment instance
            model: Model/agent instance with act(obs) method
            episode_id: Episode number (for logging)
            max_steps: Max steps per episode (None = full episode)
        
        Returns:
            (total_reward, step_logs)
        """
        obs, info = env.reset()
        if hasattr(model, 'reset'):
            model.reset()
        
        total_reward = 0.0
        step_logs = []
        step = 0
        done = False
        
        while not done:
            if max_steps and step >= max_steps:
                break
            
            # Model decides
            action = model.act(obs)
            
            # Environment steps
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # RESEARCH-CRITICAL: Log every step for audit trail
            step_log = {
                "episode": episode_id,
                "step": step,
                # State components (customize based on env)
                "obs_0": float(obs[0]) if len(obs) > 0 else None,  # SoC typically
                "obs_1": float(obs[1]) if len(obs) > 1 else None,  # Price typically
                "obs_2": float(obs[2]) if len(obs) > 2 else None,
                "obs_3": float(obs[3]) if len(obs) > 3 else None,
                # Action & reward
                "action": float(action) if isinstance(action, (int, float, np.number)) else str(action),
                "reward": float(reward),
                "cumulative_reward": float(total_reward),
                # Extra info from environment
                **{k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in info.items()}
            }
            step_logs.append(step_log)
            
            obs = obs_next
            step += 1
        
        return total_reward, step_logs
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        model_name: str,
        model_params: Dict[str, Any],
        environment_params: Dict[str, Any],
        num_episodes: int = 3,
    ) -> Dict[str, Any]:
        """
        Run complete experiment with given config.
        
        RESEARCH-CRITICAL: Produces all artifacts and returns result paths.
        
        Args:
            config: ExperimentConfig with seeds and metadata
            model_name: Name of model to use
            model_params: Model hyperparameters
            environment_params: Environment configuration
            num_episodes: Number of episodes to run
        
        Returns:
            Dict with artifact paths and summary stats
        """
        # 1. Set all random seeds (reproducibility)
        self.set_seeds(config.seed)
        
        # 2. Create environment and model
        env = self.env_factory(**environment_params)
        model = self.model_factory(model_name, model_params)
        
        # 3. Run episodes
        episode_summaries = []
        all_logs = []
        
        for ep_id in range(num_episodes):
            total_reward, step_logs = self.run_episode(env, model, ep_id)
            
            # Episode summary
            episode_summary = {
                "episode": ep_id,
                "total_reward": round(total_reward, 2),
                "num_steps": len(step_logs),
                "avg_reward_per_step": round(total_reward / len(step_logs), 2) if step_logs else 0,
                "model": model_name,
                "seed": config.seed,
            }
            episode_summaries.append(episode_summary)
            all_logs.extend(step_logs)
        
        # 4. Compute aggregate metrics
        rewards = [s["total_reward"] for s in episode_summaries]
        metrics = {
            "episodes": episode_summaries,
            "statistics": {
                "mean_reward": round(np.mean(rewards), 2),
                "std_reward": round(np.std(rewards), 2),
                "min_reward": round(np.min(rewards), 2),
                "max_reward": round(np.max(rewards), 2),
                "median_reward": round(np.median(rewards), 2),
            }
        }
        
        # 5. Save all artifacts (RESEARCH-CRITICAL)
        artifacts = self.artifact_manager.save_all_artifacts(
            experiment_id=config.experiment_id,
            config=config,
            metrics=metrics,
            summary=episode_summaries,
            logs=all_logs,
        )
        
        # 6. Return result (include paths so UI can read files)
        return {
            "experiment_id": config.experiment_id,
            "status": "completed",
            "artifacts": artifacts,
            "summary_stats": metrics["statistics"],
            "num_episodes": num_episodes,
            "model_name": model_name,
        }


def create_research_experiment_config(
    model_name: str,
    model_params: Dict[str, Any],
    environment_params: Dict[str, Any],
    seed: int = None,
    num_episodes: int = 3,
    num_steps_per_episode: int = 720,
) -> ExperimentConfig:
    """
    Helper to create reproducible experiment config.
    
    RESEARCH-CRITICAL: This is the template for all experiments.
    """
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    
    experiment_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    return ExperimentConfig(
        experiment_id=experiment_id,
        timestamp=timestamp,
        seed=seed,
        model_name=model_name,
        model_params=model_params,
        environment_params=environment_params,
        num_episodes=num_episodes,
        num_steps_per_episode=num_steps_per_episode,
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Research Experiment Runner")
    print("=" * 80)
    print()
    print("Core Principles:")
    print("  1. Reproducible: Fixed seeds at every step")
    print("  2. Auditable: Every step logged to disk")
    print("  3. Research-First: All artifacts saved (no UI state)")
    print("  4. Offline: Results computed before any UI display")
    print()
