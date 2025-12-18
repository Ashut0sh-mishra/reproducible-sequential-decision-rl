"""
Example: Using the unified agent interface with BessEnv.

This demonstrates how all baseline agents work with the same environment
and can be easily compared.
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.bess_env import BessEnv
from agents.base_agent import RandomAgent, RuleBasedThresholdAgent, GreedyInstantaneousProfitAgent


def run_single_episode(agent, env, episode_id: int = 0, verbose: bool = True):
    """
    Run one episode with a given agent and return summary statistics.
    
    Args:
        agent: BaseAgent subclass instance
        env: BessEnv instance
        episode_id: Episode number (for logging)
        verbose: Print step-by-step output
    
    Returns:
        dict with episode summary
    """
    obs, info = env.reset()
    agent.reset()
    
    total_reward = 0.0
    step = 0
    actions = []
    socs = []
    prices = []
    
    done = False
    while not done:
        # Agent decides
        action = agent.act(obs)
        
        # Environment steps
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track metrics
        total_reward += reward
        actions.append(float(action[0]))
        socs.append(float(obs[0]))
        prices.append(float(obs[1]))
        
        if verbose and step % 24 == 0:  # Print daily summary
            print(
                f"  [Day {step//24}, Hour {step%24:2d}] "
                f"SoC: {obs[0]:5.1%} | Price: €{obs[1]:6.2f}/MWh | "
                f"Action: {action[0]:+6.2f} kW | Reward: €{reward:7.2f} | "
                f"Cumulative: €{total_reward:8.2f}"
            )
        
        obs = obs_next
        step += 1
    
    # Classify actions
    n_charge = sum(1 for a in actions if a > 0.1)
    n_discharge = sum(1 for a in actions if a < -0.1)
    n_idle = sum(1 for a in actions if abs(a) <= 0.1)
    
    return {
        "agent_name": agent.name,
        "episode_id": episode_id,
        "total_reward": total_reward,
        "avg_reward_per_step": total_reward / step if step > 0 else 0,
        "num_steps": step,
        "final_soc": obs[0],
        "avg_price": np.mean(prices),
        "price_std": np.std(prices),
        "n_charge_steps": n_charge,
        "n_discharge_steps": n_discharge,
        "n_idle_steps": n_idle,
        "avg_action_magnitude": np.mean(np.abs(actions)),
    }


def main():
    """Example: Compare baselines on synthetic price data."""
    
    print("\n" + "=" * 80)
    print("UNIFIED AGENT INTERFACE EXAMPLE")
    print("=" * 80)
    print()
    
    # Generate synthetic price data (30 days, hourly)
    np.random.seed(42)
    n_hours = 30 * 24
    base_price = 75.0
    prices = base_price + 20 * np.sin(2 * np.pi * np.arange(n_hours) / 24)  # Daily cycle
    prices += np.random.normal(0, 5, n_hours)  # Add noise
    prices = np.clip(prices, 20, 150)  # Realistic bounds
    
    print(f"Price data: {n_hours} hours (30 days)")
    print(f"  Range: €{prices.min():.2f} to €{prices.max():.2f}/MWh")
    print(f"  Mean: €{prices.mean():.2f}/MWh")
    print()
    
    # Create environment
    env = BessEnv(
        price_series=prices,
        battery_capacity_kwh=50.0,
        p_max_kw=10.0,
        eff=0.95,
        cost_per_full_cycle=10.0,
        soc_init=0.5,
    )
    
    # Create agents
    agents = [
        RandomAgent(p_max=10.0, seed=42),
        RuleBasedThresholdAgent(p_max=10.0, low_threshold=50, high_threshold=100),
        GreedyInstantaneousProfitAgent(p_max=10.0),
    ]
    
    print("=" * 80)
    print("COMPARING BASELINE AGENTS")
    print("=" * 80)
    print()
    
    results = []
    for agent in agents:
        print(f"\n{'-' * 80}")
        print(f"Running: {agent.name}")
        print(f"{'-' * 80}")
        
        result = run_single_episode(agent, env, episode_id=0, verbose=True)
        results.append(result)
    
    print("\n" + "=" * 80)
    print("EPISODE SUMMARY")
    print("=" * 80)
    print()
    
    # Print summary table
    print(
        f"{'Agent':<50} | {'Total EUR':>10} | {'Avg EUR/h':>8} | "
        f"{'Final SoC':>8} | {'Charge':>6} | {'Discharge':>9} | {'Idle':>4}"
    )
    print("-" * 130)
    
    for r in results:
        print(
            f"{r['agent_name']:<50} | "
            f"EUR{r['total_reward']:>8.2f} | "
            f"EUR{r['avg_reward_per_step']:>6.2f} | "
            f"{r['final_soc']:>7.1%} | "
            f"{r['n_charge_steps']:>6d} | "
            f"{r['n_discharge_steps']:>9d} | "
            f"{r['n_idle_steps']:>4d}"
        )
    
    print()
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("• Random: Baseline—should have lowest revenue (no strategy)")
    print("• Rule-Based: Simple arbitrage—exploits price thresholds")
    print("• Greedy: Myopic profit—charges/discharges instantaneously")
    print()
    print("In a full research experiment, you would:")
    print("  1. Run each agent multiple times (different price seeds)")
    print("  2. Compute mean, std-dev, and statistical significance")
    print("  3. Add PPO agent and compare all four")
    print("  4. Log all results to CSV for post-hoc analysis")
    print()


if __name__ == "__main__":
    main()
