"""
Experiment Endpoints (Research-Grade)

RESEARCH-CRITICAL: All results are saved to disk before returning to UI.
UI reads ONLY from saved files (artifacts).
No results exist in-memory or UI state.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from pathlib import Path

from app.core.research_runner import ResearchExperimentRunner, create_research_experiment_config
from app.core.experiment_artifacts import ExperimentArtifactManager
from app.agents.research_models import model_factory
from envs.bess_env import BessEnv

router = APIRouter(prefix="/experiments", tags=["Experiments (Research)"])

# RESEARCH-CRITICAL: Initialize with standard factory functions
def env_factory(**kwargs):
    """Create environment with reproducible configuration."""
    return BessEnv(**kwargs)

# Global runner instance
_runner = ResearchExperimentRunner(env_factory, model_factory)
_artifact_manager = ExperimentArtifactManager(Path("experiments"))


class ExperimentConfigRequest(BaseModel):
    """Request to run an experiment."""
    model_name: str
    model_params: Dict[str, Any]
    environment_params: Dict[str, Any]
    seed: int = None
    num_episodes: int = 3


@router.post("/run-experiment")
def run_experiment(req: ExperimentConfigRequest):
    """
    Run a complete experiment and save all artifacts.
    
    RESEARCH-CRITICAL: Results saved to disk BEFORE response.
    Returns artifact paths (UI reads from files).
    """
    try:
        # Create reproducible config
        config = create_research_experiment_config(
            model_name=req.model_name,
            model_params=req.model_params,
            environment_params=req.environment_params,
            seed=req.seed,
            num_episodes=req.num_episodes,
        )
        
        # Run experiment (saves artifacts internally)
        result = _runner.run_experiment(
            config=config,
            model_name=req.model_name,
            model_params=req.model_params,
            environment_params=req.environment_params,
            num_episodes=req.num_episodes,
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiment/{experiment_id}")
def get_experiment_results(experiment_id: str):
    """
    Retrieve results from saved artifacts.
    
    RESEARCH-CRITICAL: Read-only access to disk artifacts.
    """
    try:
        config = _artifact_manager.load_config(experiment_id)
        summary = _artifact_manager.load_summary(experiment_id)
        
        # Parse summary for stats
        if summary:
            rewards = [float(row.get("total_reward", 0)) for row in summary]
            stats = {
                "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
                "std_reward": (sum((x - sum(rewards)/len(rewards))**2 for x in rewards) / len(rewards))**0.5 if len(rewards) > 1 else 0,
                "min_reward": min(rewards) if rewards else 0,
                "max_reward": max(rewards) if rewards else 0,
            }
        else:
            stats = {}
        
        return {
            "experiment_id": experiment_id,
            "config": {
                "seed": config.seed,
                "model": config.model_name,
                "timestamp": config.timestamp,
            },
            "summary": summary,
            "stats": stats,
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")


@router.get("/experiment/{experiment_id}/logs")
def get_experiment_logs(experiment_id: str, limit: int = 100):
    """
    Retrieve step-by-step logs (for detailed inspection).
    
    RESEARCH-CRITICAL: Complete audit trail of experiment.
    """
    try:
        logs = _artifact_manager.load_logs(experiment_id)
        # Optional: limit for UI responsiveness
        if limit and len(logs) > limit:
            logs = logs[-limit:]  # Return last N steps
        return {
            "experiment_id": experiment_id,
            "logs": logs,
            "total_steps": len(_artifact_manager.load_logs(experiment_id)),
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")


@router.get("/experiments/list")
def list_experiments():
    """
    List all completed experiments.
    
    RESEARCH-CRITICAL: Enumerate saved artifacts only.
    """
    experiments = _artifact_manager.list_experiments()
    return {
        "experiments": experiments,
        "count": len(experiments),
    }
