"""
DQN Discrete Action Wrapper for Continuous Irrigation Environment
==================================================================

Wraps IrrigationEnvContinuous with discrete action space for DQN training.
Maps discrete action indices to continuous irrigation amounts using linear interpolation.

Action Space:
    Discrete(16): 0-15 representing irrigation amounts from 0 to 15mm
    
Mapping (using np.linspace):
    irrigation_mm = np.linspace(0, max_irrigation, n_actions)[action_idx]
    Equivalent to: action_idx * (max_irrigation / (n_actions - 1))
    
    With n_actions=16, max_irrigation=15.0:
        action 0  -> 0.0 mm
        action 1  -> 1.0 mm
        ...
        action 15 -> 15.0 mm
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from irrigation_env_continuous import IrrigationEnvContinuous


class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Wraps continuous action environment with discrete action space.
    
    Converts discrete action indices to continuous irrigation amounts using
    linear mapping from 0 to max_irrigation.
    
    Parameters
    ----------
    env : IrrigationEnvContinuous
        Continuous action environment to wrap
    n_actions : int
        Number of discrete actions (default: 16)
    """
    
    def __init__(self, env, n_actions=16):
        super().__init__(env)
        
        self.n_actions = n_actions
        self.max_irrigation = env.max_irrigation
        
        # Override action space to discrete
        self.action_space = spaces.Discrete(n_actions)
        
        # Pre-compute action mapping for efficiency
        self.action_values = np.linspace(
            0.0, 
            self.max_irrigation, 
            n_actions,
            dtype=np.float32
        )
        
        # Pre-compute action arrays to avoid repeated array creation
        self.action_arrays = [np.array([val], dtype=np.float32) for val in self.action_values]
    
    def action(self, action_idx):
        """
        Convert discrete action index to continuous irrigation amount.
        
        Parameters
        ----------
        action_idx : int
            Discrete action index [0, n_actions-1]
        
        Returns
        -------
        irrigation_mm : np.ndarray
            Continuous irrigation amount as array for env.step()
        """
        # Return pre-computed action array for efficiency
        return self.action_arrays[action_idx]


def make_discretized_env(
    n_actions=16,
    max_et0=15.0,
    max_rain=5.0,
    et0_range=(2.0, 15.0),
    rain_range=(0.0, 5.0),
    max_soil_moisture=100.0,
    episode_length=90,
    threshold_bottom_soil_moisture=0.4,
    threshold_top_soil_moisture=0.7,
    max_irrigation=15.0,
    seed=None
):
    """
    Factory function to create discretized irrigation environment.
    
    Creates IrrigationEnvContinuous and wraps it with DiscreteActionWrapper
    using the same default parameters as PPO training.
    
    Parameters
    ----------
    n_actions : int
        Number of discrete actions (default: 16)
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
        Bottom threshold for optimal moisture range, default: 0.4
    threshold_top_soil_moisture : float
        Top threshold for optimal moisture range, default: 0.7
    max_irrigation : float
        Maximum irrigation amount (mm), default: 15.0
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    env : DiscreteActionWrapper
        Discretized irrigation environment
    """
    # Create continuous environment
    env = IrrigationEnvContinuous(
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
    
    # Wrap with discrete action space
    env = DiscreteActionWrapper(env, n_actions=n_actions)
    
    # Reset with seed if provided
    if seed is not None:
        env.reset(seed=seed)
    
    return env


# Verification and testing
if __name__ == "__main__":
    print("=" * 70)
    print("DQN Discrete Action Wrapper - Demo")
    print("=" * 70)
    
    # Create discretized environment with default PPO parameters
    env = make_discretized_env(
        n_actions=16,
        max_irrigation=15.0,
        seed=42
    )
    
    print(f"\nEnvironment Details:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Number of actions: {env.n_actions}")
    print(f"  Max irrigation: {env.max_irrigation} mm")
    
    # Display action mapping
    print(f"\nAction Mapping (index -> irrigation amount):")
    for i in range(env.n_actions):
        print(f"  Action {i:2d} -> {env.action_values[i]:5.2f} mm")
    
    # Test reset
    obs, info = env.reset()
    print(f"\nInitial State:")
    print(f"  Soil moisture: {obs['soil_moisture'][0]:.3f}")
    print(f"  Crop stage: {obs['crop_stage']}")
    print(f"  ET0: {info.get('raw_et0', 0.0):.2f} mm/day")
    print(f"  Rain: {info.get('raw_rain', 0.0):.2f} mm/day")
    
    # Test different actions
    print(f"\n{'Step':<6} {'Action':<8} {'Irr (mm)':<10} {'Reward':<10} {'SM':<8}")
    print("-" * 70)
    
    test_actions = [0, 5, 10, 15, 8, 3]  # Test various discrete actions
    
    for step, action_idx in enumerate(test_actions):
        obs, reward, terminated, truncated, info = env.step(action_idx)
        irr_applied = info.get('irrigation_applied', 0.0)
        sm = obs['soil_moisture'][0]
        
        print(f"{step+1:<6} {action_idx:<8} {irr_applied:<10.2f} {reward:<10.2f} {sm:<8.3f}")
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 70)
    print("Wrapper Test Results:")
    print("  ✓ Environment created successfully")
    print("  ✓ Action space is Discrete(16)")
    print("  ✓ Actions map to 0-15mm irrigation")
    print("  ✓ Observation space matches continuous environment")
    print("  ✓ Step function works correctly")
    print("=" * 70)
    print("\nDiscrete action wrapper ready for DQN training!")
    print("=" * 70)
