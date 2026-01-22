"""
Discretization Wrapper for DQN Training
========================================

Wraps IrrigationEnvContinuous to provide a discrete action space for DQN,
while maintaining identical environment dynamics and reward function.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from irrigation_env_continuous import IrrigationEnvContinuous


class DiscretizedIrrigationEnv(gym.ActionWrapper):
    """
    Wrapper that converts continuous irrigation actions to discrete.
    
    Maps discrete action indices {0, 1, ..., n_actions-1} to continuous 
    irrigation amounts [0, 1, ..., n_actions-1] mm.
    
    Parameters
    ----------
    env : IrrigationEnvContinuous
        The continuous action environment to wrap
    n_actions : int
        Number of discrete actions (default: 16 for [0, 1, ..., 15] mm)
    
    Example
    -------
    >>> env = IrrigationEnvContinuous(max_irrigation=15.0)
    >>> discrete_env = DiscretizedIrrigationEnv(env, n_actions=16)
    >>> obs, _ = discrete_env.reset()
    >>> action = 10  # Apply 10 mm irrigation
    >>> obs, reward, terminated, truncated, info = discrete_env.step(action)
    """
    
    def __init__(self, env, n_actions=16):
        """
        Initialize discretized wrapper.
        
        Parameters
        ----------
        env : IrrigationEnvContinuous
            Continuous irrigation environment
        n_actions : int
            Number of discrete actions (default: 16)
        """
        super().__init__(env)
        self.n_actions = n_actions
        
        # Override action space to discrete
        self.action_space = spaces.Discrete(n_actions)
        
        # Map discrete actions to continuous values
        # Actions {0, 1, 2, ..., 15} -> {0.0, 1.0, 2.0, ..., 15.0} mm
        self.action_map = np.arange(n_actions, dtype=np.float32)
    
    def action(self, action):
        """
        Convert discrete action to continuous irrigation amount.
        
        Parameters
        ----------
        action : int
            Discrete action index [0, n_actions-1]
        
        Returns
        -------
        continuous_action : np.ndarray
            Continuous irrigation amount as array [value]
        """
        # Map discrete action to continuous value
        continuous_value = self.action_map[action]
        # Return as array for continuous environment
        return np.array([continuous_value], dtype=np.float32)


def make_discretized_env(
    max_et0=15.0,
    max_rain=5.0,
    et0_range=(2.0, 15.0),
    rain_range=(0.0, 5.0),
    max_soil_moisture=100.0,
    episode_length=90,
    threshold_bottom_soil_moisture=0.4,
    threshold_top_soil_moisture=0.7,
    max_irrigation=15.0,
    n_actions=16,
    seed=None
):
    """
    Factory function to create discretized irrigation environment for DQN.
    
    Creates IrrigationEnvContinuous wrapped with DiscretizedIrrigationEnv,
    using the same default parameters as PPO training for fair comparison.
    
    Parameters
    ----------
    max_et0 : float
        Maximum ET0 for normalization (mm/day), default: 15.0
    max_rain : float
        Maximum rainfall for normalization (mm/day), default: 5.0
    et0_range : tuple
        (min, max) range for sampling ET0 (mm/day), default: (2.0, 15.0)
    rain_range : tuple
        (min, max) range for sampling rainfall (mm/day), default: (0.0, 5.0)
    max_soil_moisture : float
        Maximum soil moisture capacity (mm), default: 100.0
    episode_length : int
        Episode length in days, default: 90
    threshold_bottom_soil_moisture : float
        Bottom threshold for optimal moisture [0, 1], default: 0.4
    threshold_top_soil_moisture : float
        Top threshold for optimal moisture [0, 1], default: 0.7
    max_irrigation : float
        Maximum irrigation amount (mm), default: 15.0
    n_actions : int
        Number of discrete actions, default: 16 (for 0-15 mm)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    env : DiscretizedIrrigationEnv
        Discretized irrigation environment ready for DQN training
    """
    # Create continuous environment
    continuous_env = IrrigationEnvContinuous(
        max_et0=max_et0,
        max_rain=max_rain,
        et0_range=et0_range,
        rain_range=rain_range,
        max_soil_moisture=max_soil_moisture,
        episode_length=episode_length,
        threshold_bottom_soil_moisture=threshold_bottom_soil_moisture,
        threshold_top_soil_moisture=threshold_top_soil_moisture,
        max_irrigation=max_irrigation
    )
    
    # Wrap with discretization
    discrete_env = DiscretizedIrrigationEnv(continuous_env, n_actions=n_actions)
    
    # Set seed if provided
    if seed is not None:
        discrete_env.reset(seed=seed)
    
    return discrete_env


# Example usage and verification
if __name__ == "__main__":
    print("=" * 70)
    print("Discretized Irrigation Environment - Demo")
    print("=" * 70)
    
    # Create discretized environment
    env = make_discretized_env(seed=42)
    
    print(f"\nAction space: {env.action_space}")
    print(f"Number of discrete actions: {env.n_actions}")
    print(f"Action mapping: {env.action_map}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial state:")
    print(f"  Soil moisture: {obs['soil_moisture'][0]:.3f}")
    print(f"  Crop stage: {obs['crop_stage']}")
    print(f"  ET₀: {info['raw_et0']:.2f} mm/day")
    print(f"  Rain: {info['raw_rain']:.2f} mm/day")
    
    # Test different discrete actions
    print(f"\n{'Step':<6} {'Action (idx)':<14} {'Mapped (mm)':<12} {'Reward':<10} {'Soil':<8}")
    print("-" * 70)
    
    test_actions = [0, 5, 10, 15, 8, 3]  # Discrete action indices
    
    for i, action_idx in enumerate(test_actions):
        obs, reward, terminated, truncated, info = env.step(action_idx)
        mapped_value = env.action_map[action_idx]
        
        print(f"{i+1:<6} {action_idx:<14} {mapped_value:<12.1f} {reward:<10.2f} {obs['soil_moisture'][0]:<8.3f}")
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 70)
    print("Discretized environment ready for DQN training!")
    print("=" * 70)
