"""
Environment Wrappers and Utilities for Irrigation Scheduling
============================================================

Provides:
1. DiscreteActionWrapper - Wraps continuous environment for DQN training
2. Environment factory functions - Create environments with monitoring and vectorization

Merged from: ppo_env.py + dqn_wrapper.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from src.envs.irrigation_env import IrrigationEnv
from src.envs.irrigation_env_continuous import IrrigationEnvContinuous


# ============================================================================
# DQN Discrete Action Wrapper
# ============================================================================

class DiscreteActionWrapper(gym.ActionWrapper):
    """
    Wraps continuous action environment with discrete action space.
    
    Converts discrete action indices to continuous irrigation amounts using
    linear mapping from 0 to max_irrigation.
    
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


# ============================================================================
# Environment Factory Functions
# ============================================================================

def make_irrigation_env(
    max_et0=8.0,
    max_rain=50.0,
    et0_range=(2.0, 8.0),
    rain_range=(0.0, 0.8),
    max_soil_moisture=320.0,
    episode_length=90,
    threshold_bottom_soil_moisture=0.4,
    threshold_top_soil_moisture=0.7,
    seed=None,
    monitor_filename=None,
    continuous=False,
    max_irrigation=30.0
):
    """
    Create an IrrigationEnv instance with optional monitoring.
    
    Parameters
    ----------
    max_et0 : float
        Maximum ET₀ for normalization (mm/day)
    max_rain : float
        Maximum rainfall for normalization (mm/day)
    et0_range : tuple
        (min, max) range for sampling ET₀ (mm/day)
    rain_range : tuple
        (min, max) range for sampling rainfall (mm/day)
    max_soil_moisture :  float
        Maximum soil water capacity (mm)
    episode_length : int
        Number of days per episode
    threshold_bottom_soil_moisture : float
        Lower threshold for optimal moisture [0, 1]
    threshold_top_soil_moisture : float
        Upper threshold for optimal moisture [0, 1]
    seed : int, optional
        Random seed for reproducibility
    monitor_filename : str, optional
        If provided, wraps env with Monitor and saves logs to this path
    continuous : bool, optional
        If True, creates IrrigationEnvContinuous with continuous action space.
        If False (default), creates discrete IrrigationEnv. Default: False
    max_irrigation : float, optional
        Maximum irrigation amount (mm) for continuous action space.
        Only used when continuous=True. Default: 30.0
    
    Returns
    -------
    env : gym.Env
        Irrigation environment (optionally wrapped with Monitor)
    """
    if continuous:
        # Create continuous action space environment
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
    else:
        # Create discrete action space environment (default)
        env = IrrigationEnv(
            max_et0=max_et0,
            max_rain=max_rain,
            et0_range=et0_range,
            rain_range=rain_range,
            max_soil_moisture=max_soil_moisture,
            episode_length=episode_length,
            threshold_bottom_soil_moisture=threshold_bottom_soil_moisture,
            threshold_top_soil_moisture=threshold_top_soil_moisture
        )
    
    if seed is not None:
        env.reset(seed=seed)
    
    if monitor_filename is not None:
        env = Monitor(env, monitor_filename)
    
    return env


def make_vec_env(
    n_envs=1,
    seed=None,
    monitor_dir=None,
    **env_kwargs
):
    """
    Create vectorized environment for parallel training.
    
    Parameters
    ----------
    n_envs : int
        Number of parallel environments
    seed : int, optional
        Base random seed (each env gets seed + i)
    monitor_dir : str, optional
        Directory to save Monitor logs for each environment
    **env_kwargs
        Additional arguments passed to make_irrigation_env
    
    Returns
    -------
    vec_env : DummyVecEnv
        Vectorized environment
    """
    def _make_env(rank):
        def _init():
            env_seed = None if seed is None else seed + rank
            monitor_file = None if monitor_dir is None else f"{monitor_dir}/env_{rank}"
            return make_irrigation_env(seed=env_seed, monitor_filename=monitor_file, **env_kwargs)
        return _init
    
    return DummyVecEnv([_make_env(i) for i in range(n_envs)])


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
    Factory function to create discretized irrigation environment for DQN.
    
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


def verify_env_compatibility(env=None, verbose=True):
    """
    Verify that environment is compatible with stable-baselines3.
    
    Parameters
    ----------
    env : gym.Env, optional
        Environment to check.  If None, creates default IrrigationEnv
    verbose : bool
        Whether to print results
    
    Returns
    -------
    compatible : bool
        True if environment passes all checks
    """
    if env is None:
        env = make_irrigation_env()
    
    try:
        check_env(env, warn=True)
        if verbose:
            print("✓ Environment is SB3-compatible")
        return True
    except Exception as e: 
        if verbose:
            print(f"✗ Environment compatibility check failed: {e}")
        return False


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__": 
    print("=" * 70)
    print("Testing Environment Wrappers and Utilities")
    print("=" * 70)
    
    # Test 1: Basic environment creation
    print("\n1. Testing basic environment creation...")
    env = make_irrigation_env(seed=42)
    print(f"✓ Created environment: {env}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test 2: Compatibility check
    print("\n2. Testing SB3 compatibility...")
    verify_env_compatibility(env)
    
    # Test 3: Vectorized environment
    print("\n3. Testing vectorized environment...")
    vec_env = make_vec_env(n_envs=4, seed=42)
    print(f"✓ Created vectorized environment: {vec_env}")
    
    # Test 4: Discretized environment for DQN
    print("\n4. Testing discretized environment for DQN...")
    dqn_env = make_discretized_env(n_actions=16, max_irrigation=15.0, seed=42)
    print(f"✓ Created discretized environment")
    print(f"  Action space: {dqn_env.action_space}")
    print(f"  Number of actions: {dqn_env.n_actions}")
    print(f"  Max irrigation: {dqn_env.max_irrigation} mm")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
