# Unified Agent Interface – Design Document

## Overview

All agents (baselines, RL, optimization) now share a single, minimal interface. This enables:
- ✅ Fair comparison across agent types
- ✅ Easy integration with experiment runner
- ✅ Reproducible research (all agents follow same contract)
- ✅ Extensibility (add new agent types without modifying runner)

---

## The BaseAgent Interface

**Three methods, one property:**

```python
class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent identifier."""
        pass

    @abstractmethod
    def reset(self):
        """Reset agent state for new episode."""
        pass

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """Choose action given current observation."""
        pass
```

### Method Details

#### `name: str` (property)
- **Purpose:** Identify agent in logs and reports
- **Example:** `"rule_based(low=50, high=100)"`, `"ppo_trained"`, `"random_agent"`
- **Why:** Essential for reproducible result tracking

#### `reset()`
- **Purpose:** Prepare agent for new episode
- **Called:** Once per episode, after `env.reset()`
- **Typical use:** Clear internal state, reset counters
- **For stateless agents (rule-based):** Can be a no-op
- **For stateful agents (RL with memory):** Reset hidden state, recurrent buffers

#### `act(observation) -> action`
- **Input:** `np.ndarray([SoC, price_now, price_next, hour_of_day], dtype=np.float32)`
- **Output:** `np.ndarray([power_kw], dtype=np.float32)`
  - Positive = charge (kW)
  - Negative = discharge (kW)
- **Contract:** Must return shape `(1,)` for consistency with environment

---

## Implemented Baseline Agents

### 1. RandomAgent
**What:** Samples uniformly from action space `[-p_max, +p_max]`

**When to use:** Sanity check—if RL/baselines don't beat random, something is wrong.

**Example:**
```python
agent = RandomAgent(p_max=10.0, seed=42)
agent.reset()
action = agent.act(obs)  # → random power in [-10, +10] kW
```

---

### 2. RuleBasedThresholdAgent
**What:** Simple arbitrage rule (price-based)

**Logic:**
```
if price < low_threshold and SoC < 0.9:
    charge at full power
elif price > high_threshold and SoC > 0.1:
    discharge at full power
else:
    idle (zero power)
```

**Parameters:**
- `low_threshold`: Buy price (€/MWh), default 50
- `high_threshold`: Sell price (€/MWh), default 100

**Why it's useful:**
- ✅ Interpretable (anyone can understand the logic)
- ✅ Common in real BESS operations (manual control)
- ✅ Tunable baseline (sweep thresholds for hyperparameter sensitivity)

**Example:**
```python
agent = RuleBasedThresholdAgent(p_max=10.0, low_threshold=40, high_threshold=120)
agent.reset()
action = agent.act(obs)
```

---

### 3. GreedyInstantaneousProfitAgent
**What:** Maximize profit in the current timestep

**Logic:**
```
if price < median:
    charge (if room)
elif price > median:
    discharge (if room)
else:
    idle
```

**Parameters:**
- `price_percentile`: Percentile for "cheap" vs. "expensive" (default 50 = median)

**Why it's useful:**
- ✅ Simpler than threshold-based (no manual parameter tuning)
- ✅ Myopic policy (no lookahead)—captures limits of short-term optimization

---

## How to Add a New Baseline

Example: **MPC (Model Predictive Control)**

```python
from agents.base_agent import BaseAgent
import cvxpy as cp
import numpy as np

class MPCAgent(BaseAgent):
    """
    MPC baseline: solves 24-hour lookahead optimization.
    Assumes perfect price forecast (unrealistic but useful upper bound).
    """
    
    def __init__(self, p_max=10.0, capacity=50.0, eff=0.95):
        self.p_max = p_max
        self.capacity = capacity
        self.eff = eff
        self._name = "mpc_lookahead_24h"
        self.future_prices = None
        self.lookahead_plan = None
        self.step_idx = 0
    
    @property
    def name(self):
        return self._name
    
    def reset(self):
        self.step_idx = 0
        self.lookahead_plan = None
    
    def act(self, observation):
        soc, price_now, price_next, hour = observation
        
        # On first step, solve 24-hour optimization (placeholder)
        if self.step_idx == 0:
            # Would use cvxpy to solve:
            # max sum(revenue_t) subject to SoC dynamics, power limits
            self.lookahead_plan = self._solve_lookahead(soc)
        
        # Return next action from plan
        action = self.lookahead_plan[self.step_idx % 24]
        self.step_idx += 1
        
        return np.array([action], dtype=np.float32)
    
    def _solve_lookahead(self, soc_init):
        # TODO: Implement cvxpy formulation
        # For now, return dummy plan
        return np.zeros(24)
```

Then use it like any other agent:
```python
agents = {
    "rule_based": RuleBasedThresholdAgent(),
    "random": RandomAgent(),
    "mpc": MPCAgent(),  # New!
}

runner.run_experiment(agents, num_episodes=5)
```

---

## PPO Integration (Deferred)

**Current Status:** `PPOAgentWrapper` is defined but not yet wired to training.

**Why separate?** PPO training is complex (vectorized envs, callbacks, hyperparameters). Baselines are simpler and allow us to test the comparison framework first.

**Integration Plan (Week 2):**

1. Train PPO agent using `agents/train_ppo.py` → saves to `results/model_xxxx.zip`
2. Wrap trained model:
   ```python
   from stable_baselines3 import PPO
   from agents.base_agent import PPOAgentWrapper
   
   model = PPO.load("results/model_trained.zip")
   ppo_agent = PPOAgentWrapper(model, p_max=10.0, name="ppo_trained_100k_steps")
   ```
3. Add to experiment:
   ```python
   agents = {
       "rule_based": RuleBasedThresholdAgent(),
       "ppo": ppo_agent,
   }
   runner.run_experiment(agents, num_episodes=10)
   ```

---

## Integration with Experiment Runner

The experiment runner (`experiment_runner.py`) expects agents conforming to `BaseAgent`:

```python
from experiment_runner import SimpleExperimentRunner
from agents.base_agent import RuleBasedThresholdAgent, RandomAgent

# Create agents
agents = {
    "rule_based": RuleBasedThresholdAgent(low_threshold=50, high_threshold=100),
    "random": RandomAgent(seed=42),
}

# Run experiments
runner = SimpleExperimentRunner(env, log_dir="logs")
runner.run_experiment(agents, num_episodes=3, verbose=True)

# Output:
#   logs/episode_0000_rule_based.csv
#   logs/episode_0001_rule_based.csv
#   logs/episode_0002_rule_based.csv
#   logs/episode_0000_random.csv
#   ...
#   reports/summary.csv  (aggregated comparison)
```

---

## Testing the Interface

Run the base_agent.py module directly to see all agents in action:

```bash
cd agents
python base_agent.py
```

Output:
```
========================================================================
Agent Interface Documentation
========================================================================

All agents implement the BaseAgent interface:
  - reset(): Prepare for new episode
  - act(obs): Choose action given observation
  - name: Agent identifier string

Observation format: [SoC, price_now, price_next, hour_of_day]
Action format: Scalar power (kW), positive=charge, negative=discharge

========================================================================
Example: Creating and using a baseline agent
========================================================================

Example observation: [0.5 67.5 72.3 14. ]
  SoC: 50.0%, Price: €67.50/MWh, Hour: 14

rule_based(low=50, high=100)        → action = +10.00 kW
random_agent(p_max=10.0)            → action = +3.27 kW
greedy_profit(p_max=10.0, ...)      → action =  0.00 kW
```

---

## Key Design Principles

✅ **Minimal interface:** Only 3 members (reset, act, name)
✅ **Observable:** Agent name is part of logs for traceability
✅ **Extensible:** Adding new agents = subclass BaseAgent, implement 3 methods
✅ **Testable:** Each agent can be tested independently
✅ **Comparable:** All agents use same observation/action space
✅ **Reproducible:** Seeding support (e.g., `RandomAgent(seed=42)`)

---

## What's Next?

1. ✅ Define unified interface (done)
2. ✅ Implement baselines (random, rule-based, greedy)
3. ⏳ Integrate experiment runner with BessEnv (next: tie together logging + agents)
4. ⏳ Train and wrap PPO agent
5. ⏳ Run full comparison experiment (rule vs. random vs. ppo)
6. ⏳ Generate research-grade results tables and plots

---

**Status:** Interface defined and documented. Ready for integration with experiment runner.
