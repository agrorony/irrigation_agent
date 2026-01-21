"""
Evaluation Script for Continuous PPO Irrigation Agent

Evaluates trained PPO agents on continuous action space with comprehensive metrics.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from ppo_env import make_irrigation_env


def evaluate_continuous_policy(
    model,
    env,
    n_episodes=50,
    verbose=True,
    save_logs=False,
    log_path=None
):
    """
    Evaluate trained PPO agent with comprehensive metrics.
    
    Parameters
    ----------
    model : PPO
        Trained PPO model
    env : gym.Env
        Continuous irrigation environment
    n_episodes : int
        Number of evaluation episodes
    verbose : bool
        Whether to print detailed results
    save_logs : bool
        Whether to save episode logs to CSV
    log_path : str, optional
        Path to save logs (required if save_logs=True)
    
    Returns
    -------
    results : dict
        Dictionary containing comprehensive evaluation metrics
    """
    # Storage for metrics
    episode_rewards = []
    episode_irrigations = []
    all_actions = []
    all_soil_moistures = []
    optimal_steps = 0
    total_steps = 0
    water_stress_events = 0
    
    # Episode-level storage for CSV
    if save_logs:
        episode_data = []
    
    # Run evaluation episodes
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_irrigation = 0
        episode_actions = []
        episode_sm = []
        
        while not done:
            # Deterministic action selection
            action, _ = model.predict(obs, deterministic=True)
            
            # Extract scalar action
            if isinstance(action, np.ndarray):
                action_val = float(action[0])
            else:
                action_val = float(action)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            episode_irrigation += info.get('irrigation_applied', action_val)
            episode_actions.append(action_val)
            
            # Soil moisture tracking
            sm = next_obs['soil_moisture'][0]
            episode_sm.append(sm)
            all_soil_moistures.append(sm)
            
            # Check if in optimal range
            if 0.4 <= sm <= 0.7:
                optimal_steps += 1
            
            # Check for water stress
            if sm < 0.4:
                water_stress_events += 1
            
            total_steps += 1
            obs = next_obs
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_irrigations.append(episode_irrigation)
        all_actions.extend(episode_actions)
        
        if save_logs:
            episode_data.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'total_irrigation': episode_irrigation,
                'mean_action': np.mean(episode_actions),
                'max_action': np.max(episode_actions),
                'mean_soil_moisture': np.mean(episode_sm),
                'min_soil_moisture': np.min(episode_sm)
            })
    
    # Calculate statistics
    rewards_mean = np.mean(episode_rewards)
    rewards_std = np.std(episode_rewards)
    rewards_min = np.min(episode_rewards)
    rewards_max = np.max(episode_rewards)
    
    irrigation_mean = np.mean(episode_irrigations)
    irrigation_std = np.std(episode_irrigations)
    irrigation_min = np.min(episode_irrigations)
    irrigation_max = np.max(episode_irrigations)
    
    actions_array = np.array(all_actions)
    action_mean = np.mean(actions_array)
    action_std = np.std(actions_array)
    action_median = np.median(actions_array)
    action_min = np.min(actions_array)
    action_max = np.max(actions_array)
    
    # Action distribution
    zero_actions = np.sum(actions_array < 0.1)
    light_actions = np.sum((actions_array >= 0.1) & (actions_array < 10.0))
    medium_actions = np.sum((actions_array >= 10.0) & (actions_array < 20.0))
    heavy_actions = np.sum(actions_array >= 20.0)
    
    zero_pct = 100.0 * zero_actions / len(actions_array)
    light_pct = 100.0 * light_actions / len(actions_array)
    medium_pct = 100.0 * medium_actions / len(actions_array)
    heavy_pct = 100.0 * heavy_actions / len(actions_array)
    
    # Soil moisture performance
    optimal_pct = 100.0 * optimal_steps / total_steps if total_steps > 0 else 0.0
    
    # Print results
    if verbose:
        print("="*80)
        print("CONTINUOUS PPO EVALUATION RESULTS")
        print("="*80)
        print(f"Episodes: {n_episodes}")
        print()
        print("Episode Rewards:")
        print(f"  Mean ± Std:        {rewards_mean:.2f} ± {rewards_std:.2f}")
        print(f"  Min / Max:         {rewards_min:.2f} / {rewards_max:.2f}")
        print()
        print("Total Irrigation (mm/episode):")
        print(f"  Mean ± Std:        {irrigation_mean:.2f} ± {irrigation_std:.2f}")
        print(f"  Min / Max:         {irrigation_min:.2f} / {irrigation_max:.2f}")
        print()
        print("Irrigation Actions (mm/step):")
        print(f"  Mean ± Std:        {action_mean:.2f} ± {action_std:.2f}")
        print(f"  Median:            {action_median:.2f}")
        print(f"  Min / Max:         {action_min:.2f} / {action_max:.2f}")
        print()
        print("Action Distribution:")
        print(f"  Zero irrigation:   {zero_pct:.1f}% of steps")
        print(f"  Light (0-10mm):    {light_pct:.1f}% of steps")
        print(f"  Medium (10-20mm):  {medium_pct:.1f}% of steps")
        print(f"  Heavy (20-30mm):   {heavy_pct:.1f}% of steps")
        print()
        print("Soil Moisture Performance:")
        print(f"  Time in optimal range [0.4, 0.7]: {optimal_pct:.1f}%")
        print(f"  Water stress events (SM < 0.4):    {water_stress_events}")
        print("="*80)
    
    # Save logs if requested
    if save_logs and log_path is not None:
        df = pd.DataFrame(episode_data)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(log_path, index=False)
        if verbose:
            print(f"\nEpisode logs saved to: {log_path}")
    
    # Return results dictionary
    results = {
        'rewards_mean': rewards_mean,
        'rewards_std': rewards_std,
        'rewards_min': rewards_min,
        'rewards_max': rewards_max,
        'irrigation_mean': irrigation_mean,
        'irrigation_std': irrigation_std,
        'irrigation_min': irrigation_min,
        'irrigation_max': irrigation_max,
        'action_mean': action_mean,
        'action_std': action_std,
        'action_median': action_median,
        'action_min': action_min,
        'action_max': action_max,
        'zero_pct': zero_pct,
        'light_pct': light_pct,
        'medium_pct': medium_pct,
        'heavy_pct': heavy_pct,
        'optimal_pct': optimal_pct,
        'water_stress_events': water_stress_events,
        'episode_rewards': episode_rewards,
        'episode_irrigations': episode_irrigations
    }
    
    return results


def run_detailed_episode(model, env, seed=42):
    """
    Run a single episode with step-by-step visualization.
    
    Parameters
    ----------
    model : PPO
        Trained PPO model
    env : gym.Env
        Environment instance
    seed : int
        Random seed for episode
    """
    print("\n" + "="*80)
    print("DETAILED EPISODE TRACE")
    print("="*80)
    
    obs, info = env.reset(seed=seed)
    done = False
    step = 0
    total_reward = 0
    
    print(f"\n{'Step':<6} {'SM':<8} {'ET0':<8} {'Rain':<8} {'Action':<10} {'Reward':<10}")
    print("-"*80)
    
    while not done and step < 20:  # Limit to first 20 steps for visibility
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action_val = float(action[0])
        else:
            action_val = float(action)
        
        # Execute
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Print step info
        sm = obs['soil_moisture'][0]
        et0 = info.get('raw_et0', 0.0)
        rain = info.get('raw_rain', 0.0)
        
        print(f"{step+1:<6} {sm:<8.3f} {et0:<8.2f} {rain:<8.2f} {action_val:<10.2f} {reward:<10.2f}")
        
        total_reward += reward
        step += 1
        obs = next_obs
    
    if step >= 20:
        print("... (showing first 20 steps only)")
    
    # Continue until episode ends
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1
    
    print("-"*80)
    print(f"Episode completed in {step} steps")
    print(f"Total reward: {total_reward:.2f}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained continuous PPO agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument("--model-path", type=str,
                        default="models/ppo_continuous/best_model/best_model.zip",
                        help="Path to trained PPO model (.zip)")
    
    # Evaluation parameters
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation")
    
    # Output options
    parser.add_argument("--save-logs", action="store_true",
                        help="Save detailed episode logs to CSV")
    parser.add_argument("--log-path", type=str,
                        default="logs/ppo_continuous/evaluation_logs.csv",
                        help="Path to save CSV logs")
    parser.add_argument("--visualize", action="store_true",
                        help="Run detailed visualization of one episode")
    parser.add_argument("--visualize-seed", type=int, default=None,
                        help="Seed for visualization episode (default: same as seed)")
    
    # Environment parameters
    parser.add_argument("--episode-length", type=int, default=90,
                        help="Episode length (days)")
    parser.add_argument("--max-et0", type=float, default=8.0,
                        help="Maximum ET0 (mm/day)")
    parser.add_argument("--max-rain", type=float, default=50.0,
                        help="Maximum rainfall (mm/day)")
    parser.add_argument("--max-soil-moisture", type=float, default=320.0,
                        help="Maximum soil moisture (mm)")
    parser.add_argument("--max-irrigation", type=float, default=30.0,
                        help="Maximum irrigation amount (mm)")
    
    # Verbosity
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=quiet, 1=normal)")
    
    args = parser.parse_args()
    
    # Print header
    if args.verbose > 0:
        print("="*80)
        print("CONTINUOUS PPO EVALUATION")
        print("="*80)
        print(f"Model: {args.model_path}")
        print(f"Seed: {args.seed}")
        print("="*80)
    
    # Load model
    try:
        model = PPO.load(args.model_path)
        if args.verbose > 0:
            print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        exit(1)
    
    # Create evaluation environment
    env = make_irrigation_env(
        seed=args.seed,
        continuous=True,
        max_irrigation=args.max_irrigation,
        episode_length=args.episode_length,
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        max_soil_moisture=args.max_soil_moisture
    )
    
    if args.verbose > 0:
        print(f"✓ Environment created: IrrigationEnvContinuous")
        print(f"  Action space: {env.action_space}")
        print()
    
    # Run evaluation
    results = evaluate_continuous_policy(
        model=model,
        env=env,
        n_episodes=args.n_episodes,
        verbose=(args.verbose > 0),
        save_logs=args.save_logs,
        log_path=args.log_path if args.save_logs else None
    )
    
    # Run detailed episode visualization if requested
    if args.visualize:
        viz_seed = args.visualize_seed if args.visualize_seed is not None else args.seed
        run_detailed_episode(model, env, seed=viz_seed)
    
    if args.verbose > 0:
        print("\n✓ Evaluation complete")
