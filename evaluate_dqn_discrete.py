"""
Evaluation Script for DQN Discrete Irrigation Agent
====================================================

Evaluates trained DQN agents with comprehensive metrics matching PPO evaluation.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from dqn_wrapper import make_discretized_env
from dqn_agent import DQNAgent


def plot_evaluation_results(results, save_path="evaluation_plots.png"):
    """
    Create visualization of evaluation results.
    
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
    ax1.axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax1.axvline(np.median(episode_rewards), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(episode_rewards):.2f}')
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
    percentages = [results['zero_pct'], results['light_pct'], 
                   results['medium_pct'], results['heavy_pct']]
    colors_action = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    bars = ax3.bar(categories, percentages, color=colors_action, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('Action Distribution', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
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
    
    # Normalize values for visualization
    display_values = [values[0], values[1]/10, values[2], values[3]/10]
    
    bars = ax4.bar(metrics, display_values, color=colors_perf, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
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


def plot_single_episode(agent, env, seed=42, save_path="logs/dqn_discrete/episode_rollout.png"):
    """
    Plot detailed rollout of a single episode.
    
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
    # Run episode
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
        # Get action (greedy)
        action = agent.select_action(obs, epsilon=0.0)
        
        # Store current state
        steps.append(step)
        soil_moistures.append(obs['soil_moisture'][0])
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store results
        actions.append(action)
        rewards.append(reward)
        et0_values.append(info.get('raw_et0', 0.0))
        rain_values.append(info.get('raw_rain', 0.0))
        
        step += 1
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
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
    Evaluate trained DQN agent with comprehensive metrics.
    
    Parameters
    ----------
    agent : DQNAgent
        Trained DQN agent
    env : gym.Env
        Discrete irrigation environment
    n_episodes : int
        Number of evaluation episodes
    verbose : bool
        Whether to print detailed results
    save_logs : bool
        Whether to save episode logs to CSV
    log_path : str, optional
        Path to save logs
    
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
    
    # Episode-level storage
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
            # Greedy action selection
            action = agent.select_action(obs, epsilon=0.0)
            
            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track metrics
            episode_reward += reward
            # Get actual irrigation amount from action mapping
            # For default wrapper: action index directly maps to mm (0->0mm, 1->1mm, ..., 15->15mm)
            irrigation_mm = action  # Since action_map = [0, 1, 2, ..., 15] by default
            episode_irrigation += irrigation_mm
            episode_actions.append(action)
            
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
    
    actions_array = np.array(all_actions)
    action_mean = np.mean(actions_array)
    action_std = np.std(actions_array)
    
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
        print("Irrigation Actions (mm/step):")
        print(f"  Mean ± Std:        {action_mean:.2f} ± {action_std:.2f}")
        print()
        print("Action Distribution:")
        print(f"  Zero irrigation:   {zero_pct:.1f}% of steps")
        print(f"  Light (0.1-10mm):  {light_pct:.1f}% of steps")
        print(f"  Medium (10-20mm):  {medium_pct:.1f}% of steps")
        print(f"  Heavy (≥20mm):     {heavy_pct:.1f}% of steps")
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
    parser.add_argument("--model-path", type=str,
                        default="models/dqn_discrete/dqn_discrete_seed42_steps200000.pt",
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
    
    # Environment parameters
    parser.add_argument("--episode-length", type=int, default=90,
                        help="Episode length (days)")
    parser.add_argument("--max-et0", type=float, default=15.0,
                        help="Maximum ET0 (mm/day)")
    parser.add_argument("--max-rain", type=float, default=5.0,
                        help="Maximum rainfall (mm/day)")
    parser.add_argument("--et0-range", type=float, nargs=2, default=[2.0, 15.0],
                        help="ET0 range (min max)")
    parser.add_argument("--rain-range", type=float, nargs=2, default=[0.0, 5.0],
                        help="Rain range (min max)")
    parser.add_argument("--max-soil-moisture", type=float, default=100.0,
                        help="Maximum soil moisture (mm)")
    parser.add_argument("--max-irrigation", type=float, default=15.0,
                        help="Maximum irrigation amount (mm)")
    parser.add_argument("--threshold-bottom", type=float, default=0.4,
                        help="Bottom soil moisture threshold")
    parser.add_argument("--threshold-top", type=float, default=0.7,
                        help="Top soil moisture threshold")
    parser.add_argument("--n-actions", type=int, default=16,
                        help="Number of discrete actions")
    
    # Verbosity
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")
    
    args = parser.parse_args()
    
    # Print header
    if args.verbose > 0:
        print("="*80)
        print("DQN DISCRETE EVALUATION")
        print("="*80)
        print(f"Model: {args.model_path}")
        print(f"Seed: {args.seed}")
        print("="*80)
    
    # Create environment
    env = make_discretized_env(
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        et0_range=tuple(args.et0_range),
        rain_range=tuple(args.rain_range),
        max_soil_moisture=args.max_soil_moisture,
        episode_length=args.episode_length,
        threshold_bottom_soil_moisture=args.threshold_bottom,
        threshold_top_soil_moisture=args.threshold_top,
        max_irrigation=args.max_irrigation,
        n_actions=args.n_actions,
        seed=args.seed
    )
    
    if args.verbose > 0:
        print(f"✓ Environment created: DiscretizedIrrigationEnv")
        print(f"  Action space: {env.action_space}")
        print(f"  N actions: {args.n_actions}")
        print()
    
    # Load agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # State dimension: soil_moisture(1) + crop_stage_onehot(3) + rain(1) + et0(1) = 6
    STATE_DIM = 6
    
    agent = DQNAgent(
        state_dim=STATE_DIM,
        n_actions=args.n_actions,
        device=device
    )
    
    try:
        agent.load(args.model_path)
        if args.verbose > 0:
            print("✓ Model loaded successfully")
            print()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        exit(1)
    
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
