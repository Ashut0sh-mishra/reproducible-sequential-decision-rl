"""
Unified Agent Interface for Research Experiments

All agents (rule-based, random, RL, optimization) share this interface.
This enables fair comparison and easy integration with the experiment runner.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all BESS control agents.
    
    All agents must implement:
    - reset(): Prepare agent for a new episode (reset any internal state)
    - act(observation): Choose an action given current observation
    - name: Human-readable agent identifier
    
    Observation format (from BessEnv):
        [SoC (0..1), price_now (€/MWh), price_next (€/MWh), hour_of_day (0..23)]
    
    Action format (for BessEnv):
        Scalar in [-p_max, p_max] kW
        Positive = charge, Negative = discharge
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return agent identifier (e.g., 'rule_based_threshold', 'ppo_trained')."""
        pass

    @abstractmethod
    def reset(self):
        """
        Reset agent state for a new episode.
        Called at the start of each episode (env.reset()).
        """
        pass

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Choose action given current observation.
        
        Args:
            observation: np.ndarray of shape (4,)
                [SoC, price_now, price_next, hour_of_day]
        
        Returns:
            action: np.ndarray of shape (1,) 
                Scalar power in kW (typically in [-p_max, p_max])
        """
        pass


class RandomAgent(BaseAgent):
    """
    Random baseline: sample uniformly from action space.
    Useful sanity check—RL should beat this.
    """

    def __init__(self, p_max: float = 10.0, seed: int = None):
        """
        Args:
            p_max: Max power in kW (action range: [-p_max, +p_max])
            seed: Random seed for reproducibility
        """
        self.p_max = float(p_max)
        self.rng = np.random.RandomState(seed)
        self._name = f"random_agent(p_max={p_max})"

    @property
    def name(self) -> str:
        return self._name

    def reset(self):
        """No internal state to reset."""
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Sample uniformly from [-p_max, +p_max]."""
        action = self.rng.uniform(-self.p_max, self.p_max, size=(1,))
        return action.astype(np.float32)


class RuleBasedThresholdAgent(BaseAgent):
    """
    Simple rule-based baseline: charge when price is low, discharge when price is high.
    
    Logic:
    - If price < low_threshold AND SoC < 0.9  →  charge at full power
    - If price > high_threshold AND SoC > 0.1  →  discharge at full power
    - Otherwise  →  idle (zero power)
    
    Rationale: Exploits price arbitrage without learning.
    """

    def __init__(
        self,
        p_max: float = 10.0,
        low_threshold: float = 50.0,
        high_threshold: float = 100.0,
    ):
        """
        Args:
            p_max: Max power in kW
            low_threshold: Buy price threshold (€/MWh)
            high_threshold: Sell price threshold (€/MWh)
        """
        self.p_max = float(p_max)
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self._name = (
            f"rule_based(low={low_threshold}, high={high_threshold})"
        )

    @property
    def name(self) -> str:
        return self._name

    def reset(self):
        """No internal state to reset."""
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Decide: charge, discharge, or idle based on price and SoC.
        
        Args:
            observation: [SoC (0..1), price_now, price_next, hour]
        
        Returns:
            action: Scalar power (kW)
        """
        soc, price_now, price_next, hour = observation

        # Charge when price is cheap and battery not full
        if price_now < self.low_threshold and soc < 0.9:
            action = np.array([self.p_max], dtype=np.float32)

        # Discharge when price is expensive and battery not empty
        elif price_now > self.high_threshold and soc > 0.1:
            action = np.array([-self.p_max], dtype=np.float32)

        # Idle otherwise
        else:
            action = np.array([0.0], dtype=np.float32)

        return action


class GreedyInstantaneousProfitAgent(BaseAgent):
    """
    Greedy baseline: maximize instantaneous profit this step.
    
    Logic:
    - If price is below median (historically cheap) and room to charge  →  charge
    - If price is above median (historically expensive) and room to discharge  →  discharge
    - Otherwise  →  idle
    
    Simpler than threshold-based (no manual parameter tuning).
    """

    def __init__(self, p_max: float = 10.0, price_percentile: float = 50.0):
        """
        Args:
            p_max: Max power in kW
            price_percentile: Percentile for price comparison (50 = median)
        """
        self.p_max = float(p_max)
        self.price_percentile = float(price_percentile)
        self.price_threshold = None  # Will be updated with actual data
        self._name = (
            f"greedy_profit(p_max={p_max}, percentile={price_percentile})"
        )

    @property
    def name(self) -> str:
        return self._name

    def reset(self):
        """Reset price threshold (would be computed from episode data in practice)."""
        self.price_threshold = None

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Decide action based on instantaneous price vs. percentile.
        Note: In a real implementation, percentile would be computed from
        historical data or a sliding window.
        
        Args:
            observation: [SoC (0..1), price_now, price_next, hour]
        
        Returns:
            action: Scalar power (kW)
        """
        soc, price_now, price_next, hour = observation

        # Fallback: use price_now as simple threshold
        # (In practice, compute percentile from price_series)
        if price_now < 50.0 and soc < 0.9:  # Cheap → charge
            action = np.array([self.p_max], dtype=np.float32)
        elif price_now > 100.0 and soc > 0.1:  # Expensive → discharge
            action = np.array([-self.p_max], dtype=np.float32)
        else:
            action = np.array([0.0], dtype=np.float32)

        return action


# ============================================================================
# PPO WRAPPER (to be implemented later)
# ============================================================================

class PPOAgentWrapper(BaseAgent):
    """
    Wrapper around a trained PPO policy (stable-baselines3).
    
    This enables PPO models to use the same agent interface as baselines.
    Implementation postponed until PPO training is integrated.
    
    Example usage (future):
        model = PPO.load("path/to/model.zip")
        rl_agent = PPOAgentWrapper(model, p_max=10.0, name="ppo_trained")
        action = rl_agent.act(obs)
    """

    def __init__(self, policy_model, p_max: float = 10.0, name: str = "ppo_agent"):
        """
        Args:
            policy_model: Trained PPO model from stable_baselines3
            p_max: Max power (for scaling, if needed)
            name: Human-readable agent name
        """
        self.policy = policy_model
        self.p_max = float(p_max)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reset(self):
        """PPO has no episodic state to reset (stateless policy)."""
        pass

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Query trained PPO policy for action.
        
        Args:
            observation: np.ndarray of shape (4,)
        
        Returns:
            action: np.ndarray of shape (1,)
        """
        # Ensure observation is the right shape and type
        obs = np.array(observation, dtype=np.float32).reshape(1, -1)
        
        # Get action from policy (deterministic or stochastic)
        action, _ = self.policy.predict(obs, deterministic=True)
        
        return action.astype(np.float32)


if __name__ == "__main__":
    print("=" * 70)
    print("Agent Interface Documentation")
    print("=" * 70)
    print()
    print("All agents implement the BaseAgent interface:")
    print("  - reset(): Prepare for new episode")
    print("  - act(obs): Choose action given observation")
    print("  - name: Agent identifier string")
    print()
    print("Observation format: [SoC, price_now, price_next, hour_of_day]")
    print("Action format: Scalar power (kW), positive=charge, negative=discharge")
    print()
    print("=" * 70)
    print("Example: Creating and using a baseline agent")
    print("=" * 70)
    print()

    # Create example agents
    baseline_rule = RuleBasedThresholdAgent(p_max=10.0, low_threshold=50, high_threshold=100)
    baseline_random = RandomAgent(p_max=10.0, seed=42)
    baseline_greedy = GreedyInstantaneousProfitAgent(p_max=10.0)

    # Example observation from environment
    example_obs = np.array([0.5, 67.5, 72.3, 14.0], dtype=np.float32)
    print(f"Example observation: {example_obs}")
    print(f"  SoC: {example_obs[0]:.1%}, Price: €{example_obs[1]:.2f}/MWh, Hour: {int(example_obs[3])}")
    print()

    for agent in [baseline_rule, baseline_random, baseline_greedy]:
        agent.reset()
        action = agent.act(example_obs)
        print(f"{agent.name:40s} → action = {action[0]:+7.2f} kW")

    print()
    print("=" * 70)
    print("PPO Wrapper (to be populated with trained model)")
    print("=" * 70)
    print()
    print("After training, wrap PPO model:")
    print("  model = PPO.load('path/to/trained_ppo.zip')")
    print("  ppo_agent = PPOAgentWrapper(model, p_max=10.0, name='ppo_trained')")
    print("  action = ppo_agent.act(obs)")
    print()
    print("PPO agent will then be comparable with baselines in the same experiment.")
