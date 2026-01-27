"""
Evaluation Utilities for Irrigation Scheduling
==============================================

Evaluates trained PPO and DQN agents.

Merged from: evaluate_ppo.py + evaluate_ppo_continuous.py + evaluate_dqn_discrete.py
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from src.envs.wrappers import make_irrigation_env
from src.utils.rollout import run_ppo_rollout


def evaluate_ppo(model, env, n_episodes=50, verbose=True):
    """
    Evaluate trained PPO agent with deterministic policy.
    
    Matches DQN evaluation format for direct comparison.
    
    Parameters
    ----------
    model : PPO
        Trained PPO model
    env : gym. Env
        Environment instance
    n_episodes : int
        Number of evaluation episodes
    verbose : bool
        Whether to print results
    
    Returns
    -------
    avg_return : float
        Average return over evaluation episodes
    returns : list
        List of returns for each episode
    """
    returns = []
    
    for episode in range(n_episodes):
        observation, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done: 
            # Deterministic action selection
            action, _ = model. predict(observation, deterministic=True)
            
            # Execute action
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
        
        returns.append(episode_return)
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    if verbose: 
        print(f"\nEvaluation over {n_episodes} episodes:")
        print(f"  Average Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"  Min Return: {np.min(returns):.2f}")
        print(f"  Max Return: {np.max(returns):.2f}")
    
    return avg_return, returns


def evaluate_from_path(
    model_path,
    n_episodes=50,
    seed=None,
    verbose=True,
    run_sample_rollout=True,
    rollout_seed=None,
    **env_kwargs
):
    """
    Load a trained PPO model and evaluate it.
    
    Parameters
    ----------
    model_path : str
        Path to saved PPO model (.zip file)
    n_episodes : int
        Number of evaluation episodes
    seed : int, optional
        Random seed for evaluation
    verbose : bool
        Whether to print results
    run_sample_rollout : bool
        If True, also runs a detailed rollout for inspection
    rollout_seed : int, optional
        Seed for sample rollout (if None, uses seed)
    **env_kwargs
        Additional arguments for environment creation
    
    Returns
    -------
    results : dict
        Dictionary containing: 
        - 'avg_return': average return
        - 'std_return': standard deviation of returns
        - 'returns':  list of episode returns
        - 'rollout_data': detailed rollout (if run_sample_rollout=True)
    """
    # Load model
    if verbose:
        print("="*80)
        print("PPO EVALUATION - Irrigation Scheduling")
        print("="*80)
        print(f"Loading model from:  {model_path}")
    
    model = PPO.load(model_path)
    
    # Create evaluation environment
    env = make_irrigation_env(seed=seed, **env_kwargs)
    
    if verbose:
        print(f"Environment:  IrrigationEnv")
        print(f"  Episode length: {env.episode_length} days")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space: {env.observation_space}")
        print("="*80)
    
    # Evaluate policy
    avg_return, returns = evaluate_ppo(model, env, n_episodes=n_episodes, verbose=verbose)
    
    results = {
        'avg_return': avg_return,
        'std_return': np.std(returns),
        'returns': returns,
        'min_return': np.min(returns),
        'max_return':  np.max(returns)
    }
    
    # Optional: run detailed rollout
    if run_sample_rollout:
        if verbose:
            print("\n" + "="*80)
            print("SAMPLE ROLLOUT (Detailed Episode Trace)")
            print("="*80)
        
        if rollout_seed is None:
            rollout_seed = seed if seed is not None else 42
        
        rollout_data = run_ppo_rollout(
            model, env, seed=rollout_seed, verbose=verbose
        )
        results['rollout_data'] = rollout_data
    
    if verbose:
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    
    # Model parameters
    parser.add_argument("model_path", type=str,
                        help="Path to trained PPO model (.zip)")
    
    # Evaluation parameters
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for evaluation")
    parser.add_argument("--rollout-seed", type=int, default=None,
                        help="Seed for sample rollout (default: same as seed)")
    parser.add_argument("--no-rollout", action="store_true",
                        help="Skip detailed rollout display")
    
    # Environment parameters
    parser.add_argument("--episode-length", type=int, default=90,
                        help="Episode length (days)")
    parser.add_argument("--max-et0", type=float, default=8.0,
                        help="Maximum ET0 (mm/day)")
    parser.add_argument("--max-rain", type=float, default=50.0,
                        help="Maximum rainfall (mm/day)")
    parser.add_argument("--max-soil-moisture", type=float, default=320.0,
                        help="Maximum soil moisture (mm)")
    
    # Output parameters
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=quiet, 1=normal)")
    
    args = parser. parse_args()
    
    # Evaluate model
    results = evaluate_from_path(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        seed=args. seed,
        verbose=(args.verbose > 0),
        run_sample_rollout=(not args.no_rollout),
        rollout_seed=args. rollout_seed,
        episode_length=args.episode_length,
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        max_soil_moisture=args.max_soil_moisture
    )
    
    # Print summary
    if args.verbose > 0:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Average Return: {results['avg_return']:.2f} ± {results['std_return']:.2f}")
        print(f"Min Return: {results['min_return']:.2f}")
        print(f"Max Return: {results['max_return']:.2f}")
        print("="*80)"""
Evaluation Script for Continuous PPO Irrigation Agent

Evaluates trained PPO agents on continuous action space with comprehensive metrics.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from ppo_env import make_irrigation_env


def plot_evaluation_results(results, save_path="evaluation_plots.png"):
    """
    Create visualization of evaluation results.
    
    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_continuous_policy
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('PPO Model Performance Evaluation', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards Distribution
    ax1 = axes[0, 0]
    episode_rewards = results['episode_rewards']
    ax1.hist(episode_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax1.axvline(np.median(episode_rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(episode_rewards):.2f}')
    ax1.set_xlabel('Episode Reward', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Episode Rewards Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Episode Rewards Over Time
    ax2 = axes[0, 1]
    episodes = range(1, len(episode_rewards) + 1)
    ax2.plot(episodes, episode_rewards, 'o-', alpha=0.6, color='blue', markersize=4)
    ax2.axhline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.fill_between(episodes, 
                      np.mean(episode_rewards) - np.std(episode_rewards),
                      np.mean(episode_rewards) + np.std(episode_rewards),
                      alpha=0.2, color='red')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Across Episodes', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action Distribution
    ax3 = axes[1, 0]
    categories = ['Zero\n(<0.1mm)', 'Light\n(0.1-10mm)', 'Medium\n(10-20mm)', 'Heavy\n(≥20mm)']
    percentages = [results['zero_pct'], results['light_pct'], results['medium_pct'], results['heavy_pct']]
    colors_action = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    bars = ax3.bar(categories, percentages, color=colors_action, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Action Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Performance Summary
    ax4 = axes[1, 1]
    metrics = ['Mean\nReward', 'Mean\nIrrigation\n(mm)', 'Optimal\nSM %', 'Stress\nEvents']
    values = [
        results['rewards_mean'],
        results['irrigation_mean'],
        results['optimal_pct'],
        results['water_stress_events']
    ]
    colors_perf = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    # Normalize values for better visualization
    display_values = [values[0], values[1]/10, values[2], values[3]/10]  # Scale for visibility
    
    bars = ax4.bar(metrics, display_values, color=colors_perf, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Value (normalized)', fontsize=12)
    ax4.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add actual value labels
    labels = [f'{values[0]:.1f}', f'{values[1]:.1f}', f'{values[2]:.1f}%', f'{int(values[3])}']
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Evaluation plots saved to: {save_path}")
    plt.close()


def plot_single_episode(model, env, seed=42, save_path="logs/ppo_continuous/episode_rollout.png", env_config=None):
    """
    Plot detailed rollout of a single episode showing soil moisture evolution.
    
    Parameters
    ----------
    model : PPO
        Trained PPO model
    env : gym.Env
        Environment instance
    seed : int
        Random seed for episode
    save_path : str
        Path to save the plot
    env_config : dict, optional
        Environment configuration parameters for display
    """
    # Run episode and collect data
    obs, info = env.reset(seed=seed)
    done = False
    
    steps = []
    soil_moistures = []
    actions = []
    rewards = []
    et0_values = []
    rain_values = []
    
    step = 0
    while not done:
        # Get action
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, np.ndarray):
            action_val = float(action[0])
        else:
            action_val = float(action)
        
        # Store current state
        steps.append(step)
        soil_moistures.append(obs['soil_moisture'][0])
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store results
        actions.append(action_val)
        rewards.append(reward)
        et0_values.append(info.get('raw_et0', 0.0))
        rain_values.append(info.get('raw_rain', 0.0))
        
        step += 1
    
    # Create figure with extra space for text
    fig = plt.figure(figsize=(15, 14))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 2, 2, 2], hspace=0.3)
    
    # Add environment info at top
    ax_info = fig.add_subplot(gs[0])
    ax_info.axis('off')
    
    # Prepare environment info text
    if env_config:
        info_text = "Environment Configuration:\n"
        info_text += f"Training: rain_range={env_config.get('train_rain_range', 'N/A')}, "
        info_text += f"max_irrigation={env_config.get('train_max_irrigation', 'N/A')} mm, "
        info_text += f"thresholds=[{env_config.get('train_threshold_bottom', 'N/A')}, {env_config.get('train_threshold_top', 'N/A')}]\n"
        info_text += f"Evaluation: rain_range={env_config.get('eval_rain_range', 'N/A')}, "
        info_text += f"max_irrigation={env_config.get('eval_max_irrigation', 'N/A')} mm, "
        info_text += f"thresholds=[{env_config.get('eval_threshold_bottom', 'N/A')}, {env_config.get('eval_threshold_top', 'N/A')}]"
    else:
        info_text = f"Environment: rain_range={(0.0, 5.0)}, max_irrigation={env.max_irrigation} mm"
    
    ax_info.text(0.5, 0.5, info_text, 
                ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.3))
    
    fig.suptitle(f'Single Episode Rollout (Seed: {seed})', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: Soil Moisture Evolution
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(steps, soil_moistures, 'b-', linewidth=2, label='Soil Moisture')
    ax1.axhline(0.4, color='orange', linestyle='--', linewidth=1.5, label='Bottom Threshold (0.4)')
    ax1.axhline(0.6, color='green', linestyle='--', linewidth=1.5, label='Top Threshold (0.6)')
    ax1.fill_between(steps, 0.4, 0.6, alpha=0.2, color='green', label='Optimal Range')
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Soil Moisture (fraction)', fontsize=12)
    ax1.set_title('Soil Moisture Evolution', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Irrigation Actions
    ax2 = fig.add_subplot(gs[2])
    ax2.bar(steps, actions, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Irrigation (mm)', fontsize=12)
    ax2.set_title('Irrigation Actions', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Environmental Conditions & Rewards
    ax3 = fig.add_subplot(gs[3])
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(steps, rewards, 'g-', linewidth=2, label='Reward', alpha=0.7)
    line2 = ax3_twin.plot(steps, et0_values, 'r--', linewidth=1.5, label='ET₀', alpha=0.7)
    line3 = ax3_twin.plot(steps, rain_values, 'b:', linewidth=1.5, label='Rain', alpha=0.7)
    
    ax3.set_xlabel('Day', fontsize=12)
    ax3.set_ylabel('Reward', fontsize=12, color='g')
    ax3_twin.set_ylabel('ET₀ / Rain (mm/day)', fontsize=12)
    ax3.set_title('Rewards and Environmental Conditions', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Episode rollout plot saved to: {save_path}")
    plt.close()


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
    parser.add_argument("--max-et0", type=float, default=15.0,
                        help="Maximum ET0 (mm/day)")
    parser.add_argument("--max-rain", type=float, default=5.0,
                        help="Maximum rainfall (mm/day)")
    parser.add_argument("--et0-range", type=float, nargs=2, default=[2.0, 15.0],
                        help="ET0 range (min max) in mm/day")
    parser.add_argument("--rain-range", type=float, nargs=2, default=[0.0, 5.0],
                        help="Rain range (min max) in mm/day")
    parser.add_argument("--max-soil-moisture", type=float, default=100.0,
                        help="Maximum soil moisture (mm)")
    parser.add_argument("--max-irrigation", type=float, default=15.0,
                        help="Maximum irrigation amount (mm)")
    parser.add_argument("--threshold-bottom", type=float, default=0.4,
                        help="Bottom soil moisture threshold")
    parser.add_argument("--threshold-top", type=float, default=0.7,
                        help="Top soil moisture threshold")
    
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
        et0_range=tuple(args.et0_range),
        rain_range=tuple(args.rain_range),
        max_soil_moisture=args.max_soil_moisture,
        threshold_bottom_soil_moisture=args.threshold_bottom,
        threshold_top_soil_moisture=args.threshold_top
    )
    
    if args.verbose > 0:
        print(f"✓ Environment created: IrrigationEnvContinuous")
        print(f"  Action space: {env.action_space}")
        print(f"  Rain range: {env.rain_range}")
        print(f"  ET0 range: {env.et0_range}")
        print(f"  Thresholds: [{env.threshold_bottom_soil_moisture}, {env.threshold_top_soil_moisture}]")
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
    
    # Generate evaluation plots
    if args.verbose > 0:
        print("\n" + "="*80)
        print("Generating evaluation visualizations...")
        print("="*80)
    
    # Prepare environment configuration info for display
    env_config = {
        'train_rain_range': '(0.0, 5.0)',  # Training default
        'train_max_irrigation': 15.0,
        'train_threshold_bottom': 0.4,
        'train_threshold_top': 0.7,
        'eval_rain_range': f'({env.rain_range[0]}, {env.rain_range[1]})',
        'eval_max_irrigation': env.max_irrigation,
        'eval_threshold_bottom': env.threshold_bottom_soil_moisture,
        'eval_threshold_top': env.threshold_top_soil_moisture
    }
    
    # Check if evaluation environment matches training
    if (env.rain_range[0] != 0.0 or env.rain_range[1] != 5.0 or 
        env.max_irrigation != 15.0 or 
        env.threshold_bottom_soil_moisture != 0.4 or 
        env.threshold_top_soil_moisture != 0.7):
        print("\n⚠ WARNING: Evaluation environment differs from training defaults!")
        print(f"  Training: rain=(0.0, 5.0), max_irr=15.0, thresholds=[0.4, 0.7]")
        print(f"  Evaluation: rain={env.rain_range}, max_irr={env.max_irrigation}, thresholds=[{env.threshold_bottom_soil_moisture}, {env.threshold_top_soil_moisture}]")
    
    plot_evaluation_results(results, save_path="logs/ppo_continuous/evaluation_performance.png")
    plot_single_episode(model, env, seed=args.seed, 
                       save_path="logs/ppo_continuous/episode_rollout.png",
                       env_config=env_config)
    
    # Run detailed episode visualization if requested
    if args.visualize:
        viz_seed = args.visualize_seed if args.visualize_seed is not None else args.seed
        run_detailed_episode(model, env, seed=viz_seed)
    
    if args.verbose > 0:
        print("\n✓ Evaluation complete")
"""
Evaluation Script for DQN Discrete Agent
=========================================

Evaluates trained DQN agents on discrete action space with comprehensive metrics
matching the PPO evaluation protocol for fair comparison.

Usage:
    python evaluate_dqn_discrete.py --model-path models/dqn_discrete/dqn_discrete_seed42_steps200000.pt --n-episodes 50
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from dqn_wrapper import make_discretized_env
from dqn_agent import DQNAgent


def plot_evaluation_results(results, save_path="logs/dqn_discrete/evaluation_performance.png"):
    """
    Create visualization of evaluation results with 2x2 subplot layout.
    
    Parameters
    ----------
    results : dict
        Results dictionary from evaluate_dqn_policy
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DQN Model Performance Evaluation', fontsize=16, fontweight='bold')
    
    # Plot 1: Episode Rewards Distribution
    ax1 = axes[0, 0]
    episode_rewards = results['episode_rewards']
    ax1.hist(episode_rewards, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax1.axvline(np.median(episode_rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(episode_rewards):.2f}')
    ax1.set_xlabel('Episode Reward', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Episode Rewards Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Episode Rewards Over Time
    ax2 = axes[0, 1]
    episodes = range(1, len(episode_rewards) + 1)
    ax2.plot(episodes, episode_rewards, 'o-', alpha=0.6, color='blue', markersize=4)
    ax2.axhline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.fill_between(episodes, 
                      np.mean(episode_rewards) - np.std(episode_rewards),
                      np.mean(episode_rewards) + np.std(episode_rewards),
                      alpha=0.2, color='red')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Across Episodes', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Action Distribution
    ax3 = axes[1, 0]
    categories = ['Zero\n(<0.1mm)', 'Light\n(0.1-7.5mm)', 'Medium\n(7.5-15mm)', 'Heavy\n(>15mm)']
    percentages = [results['zero_pct'], results['light_pct'], results['medium_pct'], results['heavy_pct']]
    colors_action = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    bars = ax3.bar(categories, percentages, color=colors_action, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Action Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Performance Summary
    ax4 = axes[1, 1]
    metrics = ['Mean\nReward', 'Mean\nIrrigation\n(mm)', 'Optimal\nSM %', 'Stress\nEvents']
    values = [
        results['rewards_mean'],
        results['irrigation_mean'],
        results['optimal_pct'],
        results['water_stress_events']
    ]
    colors_perf = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    # Normalize values for better visualization
    display_values = [values[0], values[1]/10, values[2], values[3]/10]
    
    bars = ax4.bar(metrics, display_values, color=colors_perf, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Value (normalized)', fontsize=12)
    ax4.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add actual value labels
    labels = [f'{values[0]:.1f}', f'{values[1]:.1f}', f'{values[2]:.1f}%', f'{int(values[3])}']
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Evaluation plots saved to: {save_path}")
    plt.close()


def plot_single_episode(agent, env, seed=42, save_path="logs/dqn_discrete/episode_rollout.png"):
    """
    Plot detailed rollout of a single episode showing soil moisture evolution.
    
    Parameters
    ----------
    agent : DQNAgent
        Trained DQN agent
    env : gym.Env
        Environment instance
    seed : int
        Random seed for episode
    save_path : str
        Path to save the plot
    """
    # Run episode and collect data
    obs, info = env.reset(seed=seed)
    done = False
    
    steps = []
    soil_moistures = []
    actions = []
    rewards = []
    et0_values = []
    rain_values = []
    
    step = 0
    while not done:
        # Get action
        action = agent.select_action(obs, evaluate=True)
        
        # Store current state
        steps.append(step)
        soil_moistures.append(obs['soil_moisture'][0])
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store results
        actions.append(info.get('irrigation_applied', 0.0))
        rewards.append(reward)
        et0_values.append(info.get('raw_et0', 0.0))
        rain_values.append(info.get('raw_rain', 0.0))
        
        step += 1
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'Single Episode Rollout (Seed: {seed})', fontsize=16, fontweight='bold')
    
    # Plot 1: Soil Moisture Evolution
    ax1 = axes[0]
    ax1.plot(steps, soil_moistures, 'b-', linewidth=2, label='Soil Moisture')
    ax1.axhline(0.4, color='orange', linestyle='--', linewidth=1.5, label='Bottom Threshold (0.4)')
    ax1.axhline(0.7, color='green', linestyle='--', linewidth=1.5, label='Top Threshold (0.7)')
    ax1.fill_between(steps, 0.4, 0.7, alpha=0.2, color='green', label='Optimal Range')
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Soil Moisture (fraction)', fontsize=12)
    ax1.set_title('Soil Moisture Evolution', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: Irrigation Actions
    ax2 = axes[1]
    ax2.bar(steps, actions, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Irrigation (mm)', fontsize=12)
    ax2.set_title('Irrigation Actions', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Environmental Conditions & Rewards
    ax3 = axes[2]
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(steps, rewards, 'g-', linewidth=2, label='Reward', alpha=0.7)
    line2 = ax3_twin.plot(steps, et0_values, 'r--', linewidth=1.5, label='ET₀', alpha=0.7)
    line3 = ax3_twin.plot(steps, rain_values, 'b:', linewidth=1.5, label='Rain', alpha=0.7)
    
    ax3.set_xlabel('Day', fontsize=12)
    ax3.set_ylabel('Reward', fontsize=12, color='g')
    ax3_twin.set_ylabel('ET₀ / Rain (mm/day)', fontsize=12)
    ax3.set_title('Rewards and Environmental Conditions', fontsize=13, fontweight='bold')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='best')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Episode rollout plot saved to: {save_path}")
    plt.close()


def evaluate_dqn_policy(
    agent,
    env,
    n_episodes=50,
    verbose=True,
    save_logs=False,
    log_path=None
):
    """
    Evaluate trained DQN agent with comprehensive metrics matching PPO evaluation.
    
    Parameters
    ----------
    agent : DQNAgent
        Trained DQN agent
    env : gym.Env
        Discretized irrigation environment
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
            # Greedy action selection (evaluate=True)
            action = agent.select_action(obs, evaluate=True)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            action_value = info.get('irrigation_applied', 0.0)
            episode_irrigation += action_value
            episode_actions.append(action_value)
            
            # Soil moisture tracking
            sm = next_obs['soil_moisture'][0]
            episode_sm.append(sm)
            all_soil_moistures.append(sm)
            
            # Check if in optimal range [0.4, 0.7]
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
    
    actions_array = np.array(all_actions)
    
    # Action distribution (matching PPO evaluation categories)
    # Note: DQN max is 15mm, adjusted to align with PPO's relative categorization
    # For DQN: 0-15mm range -> zero (<0.1), light (0.1-7.5), medium (7.5-15), heavy (>15, none)
    # This maintains the spirit of PPO's zero/light/medium/heavy while respecting DQN's max
    zero_actions = np.sum(actions_array < 0.1)  # <0.1mm (essentially 0)
    light_actions = np.sum((actions_array >= 0.1) & (actions_array < 7.5))  # 0.1-7.5mm
    medium_actions = np.sum((actions_array >= 7.5) & (actions_array <= 15.0))  # 7.5-15mm
    heavy_actions = np.sum(actions_array > 15.0)  # >15mm (should be 0 for DQN)
    
    total_actions = len(actions_array)
    zero_pct = 100.0 * zero_actions / total_actions
    light_pct = 100.0 * light_actions / total_actions
    medium_pct = 100.0 * medium_actions / total_actions
    heavy_pct = 100.0 * heavy_actions / total_actions
    
    # Soil moisture performance
    optimal_pct = 100.0 * optimal_steps / total_steps if total_steps > 0 else 0.0
    
    # Print results
    if verbose:
        print("="*80)
        print("DQN DISCRETE EVALUATION RESULTS")
        print("="*80)
        print(f"Episodes: {n_episodes}")
        print()
        print("Episode Rewards:")
        print(f"  Mean ± Std:        {rewards_mean:.2f} ± {rewards_std:.2f}")
        print(f"  Min / Max:         {rewards_min:.2f} / {rewards_max:.2f}")
        print()
        print("Total Irrigation (mm/episode):")
        print(f"  Mean ± Std:        {irrigation_mean:.2f} ± {irrigation_std:.2f}")
        print()
        print("Action Distribution:")
        print(f"  Zero (<0.1mm):     {zero_pct:.1f}% of steps")
        print(f"  Light (0.1-7.5mm): {light_pct:.1f}% of steps")
        print(f"  Medium (7.5-15mm): {medium_pct:.1f}% of steps")
        print(f"  Heavy (>15mm):     {heavy_pct:.1f}% of steps")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN discrete agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained DQN model (.pt)")
    
    # Evaluation parameters
    parser.add_argument("--n-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation")
    
    # Output options
    parser.add_argument("--save-logs", action="store_true",
                        help="Save detailed episode logs to CSV")
    parser.add_argument("--log-path", type=str,
                        default="logs/dqn_discrete/evaluation_logs.csv",
                        help="Path to save CSV logs")
    
    # Environment parameters (should match training)
    parser.add_argument("--n-actions", type=int, default=16,
                        help="Number of discrete actions")
    parser.add_argument("--episode-length", type=int, default=90,
                        help="Episode length (days)")
    parser.add_argument("--max-et0", type=float, default=15.0,
                        help="Maximum ET0 (mm/day)")
    parser.add_argument("--max-rain", type=float, default=5.0,
                        help="Maximum rainfall (mm/day)")
    parser.add_argument("--max-soil-moisture", type=float, default=100.0,
                        help="Maximum soil moisture (mm)")
    parser.add_argument("--max-irrigation", type=float, default=15.0,
                        help="Maximum irrigation amount (mm)")
    parser.add_argument("--threshold-bottom", type=float, default=0.4,
                        help="Bottom soil moisture threshold")
    parser.add_argument("--threshold-top", type=float, default=0.7,
                        help="Top soil moisture threshold")
    
    # Verbosity
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=quiet, 1=normal)")
    
    args = parser.parse_args()
    
    # Print header
    if args.verbose > 0:
        print("="*80)
        print("DQN DISCRETE EVALUATION")
        print("="*80)
        print(f"Model: {args.model_path}")
        print(f"Seed: {args.seed}")
        print("="*80)
    
    # Load agent
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = DQNAgent(
            state_dim=6,
            action_dim=args.n_actions,
            device=device
        )
        agent.load(args.model_path)
        if args.verbose > 0:
            print(f"✓ Model loaded successfully from {args.model_path}")
            print(f"  Device: {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        exit(1)
    
    # Create evaluation environment
    env = make_discretized_env(
        n_actions=args.n_actions,
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        et0_range=(2.0, args.max_et0),
        rain_range=(0.0, args.max_rain),
        max_soil_moisture=args.max_soil_moisture,
        episode_length=args.episode_length,
        threshold_bottom_soil_moisture=args.threshold_bottom,
        threshold_top_soil_moisture=args.threshold_top,
        max_irrigation=args.max_irrigation,
        seed=args.seed
    )
    
    if args.verbose > 0:
        print(f"✓ Environment created: DiscreteActionWrapper")
        print(f"  Action space: {env.action_space}")
        print(f"  Number of actions: {args.n_actions}")
        print(f"  Thresholds: [{args.threshold_bottom}, {args.threshold_top}]")
        print()
    
    # Run evaluation
    results = evaluate_dqn_policy(
        agent=agent,
        env=env,
        n_episodes=args.n_episodes,
        verbose=(args.verbose > 0),
        save_logs=args.save_logs,
        log_path=args.log_path if args.save_logs else None
    )
    
    # Generate evaluation plots
    if args.verbose > 0:
        print("\n" + "="*80)
        print("Generating evaluation visualizations...")
        print("="*80)
    
    plot_evaluation_results(results, save_path="logs/dqn_discrete/evaluation_performance.png")
    plot_single_episode(agent, env, seed=args.seed, 
                       save_path="logs/dqn_discrete/episode_rollout.png")
    
    if args.verbose > 0:
        print("\n✓ Evaluation complete")
