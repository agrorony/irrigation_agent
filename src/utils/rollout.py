"""
PPO Policy Rollout for Irrigation Scheduling

Runs detailed step-by-step rollouts of trained PPO policies for inspection and debugging.
"""

import numpy as np
from stable_baselines3 import PPO
from src.envs.wrappers import make_irrigation_env


def run_ppo_rollout(model, env, seed=None, max_steps=None, verbose=True):
    """
    Run a single episode with deterministic PPO policy and log detailed step information.
    
    Parameters
    ----------
    model : PPO
        Trained PPO model
    env : gym.Env
        Environment instance
    seed : int, optional
        Random seed for reproducibility
    max_steps : int, optional
        Maximum steps to run (if None, runs until episode ends)
    verbose : bool
        If True, prints formatted table to console
    
    Returns
    -------
    rollout_data : list of dict
        List of step dictionaries with keys:  day, soil_moisture, crop_stage,
        et0, rain, action, irrigation, reward
    """
    # Reset environment
    if seed is not None:
        observation, _ = env.reset(seed=seed)
    else:
        observation, _ = env.reset()
    
    done = False
    step_count = 0
    rollout_data = []
    
    # Print header
    if verbose:
        print("\n" + "="*90)
        print("PPO POLICY ROLLOUT - Deterministic Action Selection")
        print("="*90)
        print(f"{'Day': >4} {'Soil': >8} {'Stage':>6} {'ET0':>8} {'Rain':>8} {'Action':>7} {'Irrig':>8} {'Reward':>8}")
        print(f"{'': >4} {'(frac)':>8} {'':>6} {'(mm/d)':>8} {'(mm/d)':>8} {'': >7} {'(mm)':>8} {'': >8}")
        print("-"*90)
    
    # Run episode
    while not done:
        # Select deterministic action
        action, _ = model.predict(observation, deterministic=True)
        
        # Execute action
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Extract raw values from info dict
        soil_moisture = info['soil_moisture']
        crop_stage = info['crop_stage']
        et0 = info['et0']
        rain = info['rain']
        irrigation = info['irrigation']
        
        # Store step data
        step_data = {
            'day': step_count + 1,
            'soil_moisture': soil_moisture,
            'crop_stage': crop_stage,
            'et0': et0,
            'rain': rain,
            'action': int(action),
            'irrigation':  irrigation,
            'reward': reward
        }
        rollout_data.append(step_data)
        
        # Print step
        if verbose:
            print(f"{step_data['day']:4d} {step_data['soil_moisture']:8.3f} "
                  f"{step_data['crop_stage']:6d} {step_data['et0']:8.2f} "
                  f"{step_data['rain']:8.2f} {step_data['action']:7d} "
                  f"{step_data['irrigation']: 8.2f} {step_data['reward']:8.2f}")
        
        # Update observation
        observation = next_observation
        step_count += 1
        
        # Check max_steps limit
        if max_steps is not None and step_count >= max_steps:
            break
    
    if verbose:
        print("-"*90)
        total_reward = sum(d['reward'] for d in rollout_data)
        total_irrigation = sum(d['irrigation'] for d in rollout_data)
        print(f"Episode complete:  {len(rollout_data)} days")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Total irrigation: {total_irrigation:.2f} mm")
        print("="*90 + "\n")
    
    return rollout_data


def rollout_from_path(model_path, env=None, seed=None, max_steps=None, verbose=True, **env_kwargs):
    """
    Load a trained PPO model and run a rollout. 
    
    Parameters
    ----------
    model_path : str
        Path to saved PPO model (. zip file)
    env : gym.Env, optional
        Environment to use.  If None, creates default IrrigationEnv
    seed : int, optional
        Random seed
    max_steps : int, optional
        Maximum steps to run
    verbose : bool
        Whether to print rollout table
    **env_kwargs
        Additional arguments for environment creation (if env is None)
    
    Returns
    -------
    rollout_data : list of dict
        Rollout step data
    """
    # Load model
    model = PPO.load(model_path)
    
    # Create environment if not provided
    if env is None:
        env = make_irrigation_env(seed=seed, **env_kwargs)
    
    # Run rollout
    return run_ppo_rollout(model, env, seed=seed, max_steps=max_steps, verbose=verbose)


if __name__ == "__main__": 
    import argparse
    
    parser = argparse.ArgumentParser(description="Run PPO policy rollout")
    parser.add_argument("model_path", type=str,
                        help="Path to trained PPO model (.zip)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum steps (default: full episode)")
    parser.add_argument("--episode-length", type=int, default=90,
                        help="Environment episode length")
    parser.add_argument("--max-et0", type=float, default=8.0,
                        help="Maximum ET0 (mm/day)")
    parser.add_argument("--max-rain", type=float, default=50.0,
                        help="Maximum rainfall (mm/day)")
    parser.add_argument("--max-soil-moisture", type=float, default=320.0,
                        help="Maximum soil moisture (mm)")
    
    args = parser.parse_args()
    
    # Run rollout
    rollout_data = rollout_from_path(
        model_path=args.model_path,
        seed=args.seed,
        max_steps=args.max_steps,
        episode_length=args.episode_length,
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        max_soil_moisture=args.max_soil_moisture
    )
    
    print(f"\nRollout complete:  {len(rollout_data)} steps")