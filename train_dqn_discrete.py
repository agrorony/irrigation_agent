"""
Training Script for DQN Discrete Agent
=======================================

Trains DQN agent on discretized irrigation environment using experience replay
and epsilon-greedy exploration.

Usage:
    python train_dqn_discrete.py --timesteps 200000 --seed 42
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import torch
from dqn_wrapper import make_discretized_env
from dqn_agent import DQNAgent


def plot_training_results(
    episode_rewards,
    episode_steps,
    episode_irrigations,
    save_dir="logs/dqn_discrete",
    verbose=1
):
    """
    Create summary plots of training progress.
    
    Parameters
    ----------
    episode_rewards : list
        List of episode rewards
    episode_steps : list
        List of episode lengths
    episode_irrigations : list
        List of total irrigation per episode
    save_dir : str
        Directory to save plots
    verbose : int
        Verbosity level
    """
    if verbose > 0:
        print("\n" + "="*80)
        print("Creating training summary plots...")
        print("="*80)
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Summary - Discrete Irrigation', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards over Time
        ax1 = axes[0, 0]
        episodes = range(1, len(episode_rewards) + 1)
        
        # Calculate moving average
        window = min(100, len(episode_rewards) // 10)
        if window > 0:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            moving_episodes = range(window, len(episode_rewards) + 1)
        
        ax1.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
        if window > 0:
            ax1.plot(moving_episodes, moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Episode Rewards During Training', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Episode Length over Time
        ax2 = axes[0, 1]
        ax2.plot(episodes, episode_steps, alpha=0.5, color='green', label='Episode Length')
        ax2.axhline(y=np.mean(episode_steps), color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Steps', fontsize=12)
        ax2.set_title('Episode Length During Training', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Total Irrigation per Episode
        ax3 = axes[1, 0]
        ax3.plot(episodes, episode_irrigations, alpha=0.5, color='orange', label='Total Irrigation')
        if window > 0:
            moving_avg_irr = np.convolve(episode_irrigations, np.ones(window)/window, mode='valid')
            ax3.plot(moving_episodes, moving_avg_irr, color='darkred', linewidth=2, label=f'{window}-Episode Moving Avg')
        ax3.set_xlabel('Episode', fontsize=12)
        ax3.set_ylabel('Total Irrigation (mm)', fontsize=12)
        ax3.set_title('Irrigation Usage During Training', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Performance Summary (Last 100 episodes)
        ax4 = axes[1, 1]
        
        last_n = min(100, len(episode_rewards))
        recent_rewards = episode_rewards[-last_n:]
        recent_irr = episode_irrigations[-last_n:]
        
        metrics = ['Mean\nReward', 'Std\nReward', 'Mean\nIrrigation\n(mm)']
        values = [
            np.mean(recent_rewards),
            np.std(recent_rewards),
            np.mean(recent_irr)
        ]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Value', fontsize=12)
        ax4.set_title(f'Performance Summary (Last {last_n} Episodes)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}',
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        os.makedirs(save_dir, exist_ok=True)
        plot_path = f"{save_dir}/training_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if verbose > 0:
            print(f"\n✓ Training summary plot saved to: {plot_path}")
        
        plt.close()
        
    except Exception as e:
        if verbose > 0:
            print(f"⚠ Error creating plots: {e}")


def train_dqn_discrete(
    total_timesteps=200000,
    seed=42,
    learning_rate=1e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=50000,
    buffer_size=100000,
    batch_size=64,
    target_update_freq=1000,
    learning_starts=1000,
    n_actions=16,
    save_dir="models/dqn_discrete",
    log_dir="logs/dqn_discrete",
    model_name=None,
    verbose=1,
    # Environment parameters (matching PPO defaults)
    max_et0=15.0,
    max_rain=5.0,
    et0_range=(2.0, 15.0),
    rain_range=(0.0, 5.0),
    max_soil_moisture=100.0,
    episode_length=90,
    threshold_bottom_soil_moisture=0.4,
    threshold_top_soil_moisture=0.7,
    max_irrigation=15.0
):
    """
    Train DQN agent on discretized irrigation environment.
    
    Parameters
    ----------
    total_timesteps : int
        Total number of timesteps to train
    seed : int
        Random seed for reproducibility
    learning_rate : float
        Learning rate for optimizer
    gamma : float
        Discount factor
    epsilon_start : float
        Initial exploration rate
    epsilon_end : float
        Final exploration rate
    epsilon_decay_steps : int
        Steps over which to decay epsilon
    buffer_size : int
        Replay buffer capacity
    batch_size : int
        Minibatch size
    target_update_freq : int
        Frequency of target network updates
    learning_starts : int
        Number of steps before training starts
    n_actions : int
        Number of discrete actions
    save_dir : str
        Directory to save trained model
    log_dir : str
        Directory to save training logs
    model_name : str, optional
        Name for saved model
    verbose : int
        Verbosity level
    Environment parameters:
        max_et0, max_rain, et0_range, rain_range, max_soil_moisture, episode_length,
        threshold_bottom_soil_moisture, threshold_top_soil_moisture, max_irrigation
    
    Returns
    -------
    agent : DQNAgent
        Trained DQN agent
    save_path : str
        Path where model was saved
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set random seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Print configuration
    if verbose > 0:
        print("="*80)
        print("DQN TRAINING - Discrete Action Space Irrigation")
        print("="*80)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Seed: {seed}")
        print(f"\nEnvironment: IrrigationEnvContinuous + DiscreteActionWrapper")
        print(f"  Episode length: {episode_length} days")
        print(f"  Max ET0: {max_et0} mm/day")
        print(f"  Rain range: {rain_range} mm/day")
        print(f"  Max soil moisture: {max_soil_moisture} mm")
        print(f"  Max irrigation: {max_irrigation} mm")
        print(f"  Number of actions: {n_actions} (0-{max_irrigation}mm)")
        print(f"\nDQN Configuration:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gamma: {gamma}")
        print(f"  Epsilon: {epsilon_start} -> {epsilon_end} over {epsilon_decay_steps} steps")
        print(f"  Batch size: {batch_size}")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Target update freq: {target_update_freq}")
        print(f"  Learning starts: {learning_starts}")
        print("="*80)
    
    # Create environment
    env = make_discretized_env(
        n_actions=n_actions,
        max_et0=max_et0,
        max_rain=max_rain,
        et0_range=et0_range,
        rain_range=rain_range,
        max_soil_moisture=max_soil_moisture,
        episode_length=episode_length,
        threshold_bottom_soil_moisture=threshold_bottom_soil_moisture,
        threshold_top_soil_moisture=threshold_top_soil_moisture,
        max_irrigation=max_irrigation,
        seed=seed
    )
    
    # Create DQN agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DQNAgent(
        state_dim=6,
        action_dim=n_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        device=device
    )
    
    if verbose > 0:
        print(f"\n✓ Agent initialized on device: {device}")
        print(f"✓ Environment created with {n_actions} discrete actions")
        print("\nStarting training...")
        print("="*80)
    
    # Training metrics
    episode_rewards = []
    episode_steps = []
    episode_irrigations = []
    
    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_step = 0
    episode_irrigation = 0
    timestep = 0
    episode_num = 0
    
    start_time = time.time()
    
    while timestep < total_timesteps:
        # Select action
        action = agent.select_action(obs, evaluate=False)
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Track metrics
        episode_reward += reward
        episode_step += 1
        episode_irrigation += info.get('irrigation_applied', 0.0)
        
        # Store transition
        state = agent.obs_to_state(obs)
        next_state = agent.obs_to_state(next_obs)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train agent
        if timestep >= learning_starts:
            loss = agent.train_step()
        
        obs = next_obs
        timestep += 1
        
        # Episode end
        if done:
            episode_num += 1
            episode_rewards.append(episode_reward)
            episode_steps.append(episode_step)
            episode_irrigations.append(episode_irrigation)
            
            # Log every 100 episodes
            if verbose > 0 and episode_num % 100 == 0:
                elapsed_time = time.time() - start_time
                recent_rewards = episode_rewards[-100:]
                print(f"Episode {episode_num:5d} | "
                      f"Timestep {timestep:7d}/{total_timesteps} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Mean(100): {np.mean(recent_rewards):7.2f} | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Time: {elapsed_time:.0f}s")
            
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            episode_step = 0
            episode_irrigation = 0
    
    # Training complete
    total_time = time.time() - start_time
    
    if verbose > 0:
        print("="*80)
        print(f"Training complete!")
        print(f"Total episodes: {episode_num}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Timesteps per second: {total_timesteps/total_time:.1f}")
    
    # Save final model
    if model_name is None:
        model_name = f"dqn_discrete_seed{seed}_steps{total_timesteps}.pt"
    
    save_path = f"{save_dir}/{model_name}"
    agent.save(save_path)
    
    if verbose > 0:
        print(f"Model saved to: {save_path}")
        print("="*80)
    
    # Generate training summary plots (save to log_dir)
    plot_training_results(
        episode_rewards,
        episode_steps,
        episode_irrigations,
        save_dir=log_dir,
        verbose=verbose
    )
    
    # Print final statistics
    if verbose > 0:
        last_100 = min(100, len(episode_rewards))
        print(f"\nFinal Performance (Last {last_100} episodes):")
        print(f"  Mean reward: {np.mean(episode_rewards[-last_100:]):.2f} ± {np.std(episode_rewards[-last_100:]):.2f}")
        print(f"  Mean irrigation: {np.mean(episode_irrigations[-last_100:]):.2f} mm")
        print("="*80)
    
    return agent, save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN agent on discrete irrigation scheduling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total timesteps for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Initial exploration rate")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Final exploration rate")
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000,
                        help="Steps over which to decay epsilon")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--target-update-freq", type=int, default=1000,
                        help="Target network update frequency")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="Number of steps before training starts")
    
    # Environment parameters
    parser.add_argument("--n-actions", type=int, default=16,
                        help="Number of discrete actions")
    parser.add_argument("--max-et0", type=float, default=15.0,
                        help="Maximum ET0 (mm/day)")
    parser.add_argument("--max-rain", type=float, default=5.0,
                        help="Maximum rainfall (mm/day)")
    parser.add_argument("--episode-length", type=int, default=90,
                        help="Episode length (days)")
    parser.add_argument("--max-soil-moisture", type=float, default=100.0,
                        help="Maximum soil moisture capacity (mm)")
    parser.add_argument("--max-irrigation", type=float, default=15.0,
                        help="Maximum irrigation amount (mm)")
    
    # Save/log parameters
    parser.add_argument("--save-dir", type=str, default="models/dqn_discrete",
                        help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs/dqn_discrete",
                        help="Directory to save logs")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Name for saved model")
    
    # Verbosity
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=none, 1=info)")
    
    args = parser.parse_args()
    
    # Train model
    agent, save_path = train_dqn_discrete(
        total_timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        learning_starts=args.learning_starts,
        n_actions=args.n_actions,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        verbose=args.verbose,
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        episode_length=args.episode_length,
        max_soil_moisture=args.max_soil_moisture,
        max_irrigation=args.max_irrigation
    )
