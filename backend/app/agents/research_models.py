"""
Unified Model Interface for Research Comparison

RESEARCH-CRITICAL: All models (rule-based, ML, RL) use the same interface.
This ensures fair comparison and reproducibility.

Interface:
  - Model.act(observation) -> action
  - Model.reset() (optional) -> prepare for new episode

Observations: np.ndarray of environment features
Actions: Scalar or array, as appropriate for environment
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class ResearchModel(ABC):
    """
    Abstract base for all research models.
    
    Ensures all models can be evaluated uniformly.
    """
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> float:
        """
        Choose action given current observation.
        
        Args:
            observation: np.ndarray of environment features
        
        Returns:
            float: Action (scalar) appropriate for environment
        """
        pass
    
    def reset(self):
        """
        Optional: Reset model state for new episode.
        No-op for stateless models.
        """
        pass


class RuleBasedModel(ResearchModel):
    """
    Simple rule-based baseline: Price threshold arbitrage.
    
    Logic: Charge when cheap, discharge when expensive.
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.low_threshold = params.get("low_threshold", 50.0)
        self.high_threshold = params.get("high_threshold", 100.0)
        self.p_max = params.get("p_max", 10.0)
    
    def act(self, observation: np.ndarray) -> float:
        """
        obs format: [SoC (0-1), price_now, price_next, hour_of_day]
        """
        soc, price_now = observation[0], observation[1]
        
        if price_now < self.low_threshold and soc < 0.9:
            return self.p_max  # Charge
        elif price_now > self.high_threshold and soc > 0.1:
            return -self.p_max  # Discharge
        else:
            return 0.0  # Idle


class RandomModel(ResearchModel):
    """
    Random baseline: Samples uniformly from action space.
    
    Purpose: Sanity check—should be worst performer.
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.p_max = params.get("p_max", 10.0)
        self.rng = np.random.RandomState(params.get("seed", None))
    
    def act(self, observation: np.ndarray) -> float:
        return float(self.rng.uniform(-self.p_max, self.p_max))


class GreedyModel(ResearchModel):
    """
    Greedy baseline: Maximize instantaneous profit.
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.p_max = params.get("p_max", 10.0)
        self.buy_threshold = params.get("buy_threshold", 50.0)
        self.sell_threshold = params.get("sell_threshold", 100.0)
    
    def act(self, observation: np.ndarray) -> float:
        soc, price_now = observation[0], observation[1]
        
        if price_now < self.buy_threshold and soc < 0.9:
            return self.p_max
        elif price_now > self.sell_threshold and soc > 0.1:
            return -self.p_max
        else:
            return 0.0


class DummyRLModel(ResearchModel):
    """
    Placeholder for RL model (DQN from stable-baselines3).
    
    To be implemented: Load trained PPO/DQN policy.
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.p_max = params.get("p_max", 10.0)
        # TODO: Load policy from file
    
    def act(self, observation: np.ndarray) -> float:
        # Placeholder: return random action
        return float(np.random.uniform(-self.p_max, self.p_max))


def model_factory(model_name: str, params: Dict[str, Any]) -> ResearchModel:
    """
    Factory function to create model instances.
    
    Ensures all models are instantiated consistently.
    """
    models = {
        "rule_based": RuleBasedModel,
        "random": RandomModel,
        "greedy": GreedyModel,
        "dqn": DummyRLModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](params)


if __name__ == "__main__":
    print("=" * 80)
    print("Unified Research Model Interface")
    print("=" * 80)
    print()
    print("All models implement:")
    print("  - act(observation) -> action")
    print("  - reset() (optional)")
    print()
    print("Available models:")
    print("  - rule_based: Price threshold arbitrage")
    print("  - random: Uniform random baseline")
    print("  - greedy: Instantaneous profit maximization")
    print("  - dqn: Deep RL (placeholder)")
    print()
