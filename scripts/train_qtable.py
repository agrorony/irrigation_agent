"""
Q-Table Training Script for Irrigation Scheduling
=================================================

Trains tabular Q-learning agent on irrigation environment.

Usage:
    python scripts/train_qtable.py
"""

import sys
sys.path.insert(0, '.')

from src.agents.qtable import train_q_learning, extract_policy, print_policy
from src.envs.irrigation_env import IrrigationEnv

if __name__ == "__main__":
    # Create environment (dry regime configuration)
    env = IrrigationEnv(
        max_et0=8.0,
        max_rain=50.0,
        et0_range=(2.0, 8.0),
        rain_range=(0.0, 0.8),
        max_soil_moisture=320.0,
        episode_length=90
    )

    # Train Q-learning agent
    print("Training Q-learning agent...")
    Q, epsilon = train_q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.99)

    # Extract and display policy
    policy = extract_policy(Q)
    print("\nLearned policy:")
    print_policy(policy)
