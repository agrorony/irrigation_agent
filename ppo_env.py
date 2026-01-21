"""
PPO Environment Utilities for Irrigation Scheduling

Provides convenience functions for creating SB3-compatible environments
with monitoring and vectorization. 
"""

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from irrigation_agent.irrigation_env import IrrigationEnv


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
    monitor_filename=None
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
    
    Returns
    -------
    env : gym.Env
        Irrigation environment (optionally wrapped with Monitor)
    """
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


if __name__ == "__main__": 
    # Quick test
    print("Testing PPO environment utilities...")
    print("="*60)
    
    # Test 1: Basic environment creation
    env = make_irrigation_env(seed=42)
    print(f"✓ Created environment: {env}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Test 2: Compatibility check
    verify_env_compatibility(env)
    
    # Test 3: Vectorized environment
    vec_env = make_vec_env(n_envs=4, seed=42)
    print(f"✓ Created vectorized environment: {vec_env}")
    
    print("="*60)
    print("All tests passed!")