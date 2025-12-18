"""
Experiment runner with structured logging.
Trains/evaluates agents and logs trajectories to CSV/JSON.
"""

import numpy as np
from pathlib import Path
from logging_utils import EpisodeLogger, ExperimentSummary, print_step_summary


class SimpleExperimentRunner:
    """
    Run episodes with an agent, log step-by-step and summarize results.
    """

    def __init__(self, env, log_dir: str = "logs"):
        self.env = env
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_summary = ExperimentSummary()

    def run_episode(
        self,
        agent,
        episode_id: int,
        agent_name: str = "agent",
        verbose: bool = False,
        log_frequency: int = 24,  # Print every 24 steps (daily)
    ):
        """
        Run one episode and log trajectory.
        
        Args:
            agent: Controller with act(obs) method
            episode_id: Episode number for logging
            agent_name: Name to include in logs
            verbose: Print step summaries
            log_frequency: Print summary every N steps
        """
        obs, info = self.env.reset()
        logger = EpisodeLogger(output_dir=self.log_dir)
        
        done = False
        step = 0
        cumulative_reward = 0.0

        while not done:
            # Agent decides action
            action = agent.act(obs)

            # Environment step
            obs_next, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            cumulative_reward += reward

            # Extract state for logging (customize based on your env)
            price = info.get("price", obs[1])  # Adjust based on your env
            soc = obs[0]

            # Log this step
            logger.log_step(
                timestep=step,
                price=price,
                soc=soc,
                action=float(action),
                reward=reward,
                info={"step_revenue_euro": reward},
            )

            # Terminal output
            if verbose and step % log_frequency == 0:
                print_step_summary(step, price, soc, action, cumulative_reward)

            obs = obs_next
            step += 1

        # Save episode
        result = logger.save_episode(episode_id, agent_name)
        
        # Add to summary
        self.experiment_summary.add_episode_result(
            agent_name=agent_name,
            episode=episode_id,
            total_revenue=cumulative_reward,
            num_steps=step,
            extra_metrics={"final_soc": float(obs[0])},
        )

        if verbose:
            print(f"\n[EPISODE {episode_id} COMPLETE] Agent: {agent_name}")
            print(f"  Total Revenue: €{cumulative_reward:.2f}")
            print(f"  Steps: {step}")
            print(f"  Logs: {result['csv']}")

        return result

    def run_experiment(
        self,
        agents: dict,
        num_episodes: int = 10,
        verbose: bool = True,
    ):
        """
        Run multiple agents over multiple episodes.
        
        Args:
            agents: Dict of {agent_name: agent_instance}
            num_episodes: Number of episodes per agent
            verbose: Print progress
        """
        for agent_name, agent in agents.items():
            print(f"\n{'='*70}")
            print(f"Starting training for: {agent_name}")
            print(f"{'='*70}")

            for ep_id in range(num_episodes):
                self.run_episode(
                    agent=agent,
                    episode_id=ep_id,
                    agent_name=agent_name,
                    verbose=verbose,
                    log_frequency=24,
                )

        # Save aggregate summary
        summary_path = self.experiment_summary.save_summary()
        self.experiment_summary.print_summary()
        
        return summary_path


class SimpleBaselineController:
    """Naive rule-based controller: charge when price is low, discharge when high."""

    def __init__(self, low_threshold: float = 50.0, high_threshold: float = 100.0):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def act(self, obs):
        """
        obs format (customize based on env): [SoC, price_now, price_next, hour]
        """
        soc, price_now = obs[0], obs[1]

        if price_now < self.low_threshold and soc < 0.9:
            return 5.0  # Charge
        elif price_now > self.high_threshold and soc > 0.1:
            return -5.0  # Discharge
        else:
            return 0.0  # Idle


if __name__ == "__main__":
    print("Experiment runner module loaded.")
    print("Use: runner = SimpleExperimentRunner(env)")
    print("     runner.run_experiment({'baseline': controller}, num_episodes=3)")
