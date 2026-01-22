"""
DQN Training Script for Discrete Irrigation Scheduling
=======================================================

Trains vanilla DQN agent on discretized irrigation environment.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from dqn_wrapper import make_discretized_env
from dqn_agent import DQNAgent


def plot_training_summary(episode_rewards, episode_lengths, losses, save_dir, verbose=1):
    """
    Create summary plots of training progress.
    
    Parameters
    ----------
    episode_rewards : list
        Episode rewards over training
    episode_lengths : list
        Episode lengths over training
    losses : list
        Training losses (may contain None values)
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
        
        episodes = np.arange(1, len(episode_rewards) + 1)
        
        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        ax1.plot(episodes, episode_rewards, 'b-', alpha=0.3, linewidth=0.5)
        # Moving average
        window = min(100, len(episode_rewards) // 10)
        if window > 0:
            moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-episode MA')
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Episode Reward', fontsize=12)
        ax1.set_title('Episode Rewards During Training', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Episode Lengths
        ax2 = axes[0, 1]
        ax2.plot(episodes, episode_lengths, 'g-', alpha=0.3, linewidth=0.5)
        if window > 0:
            moving_avg_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(episodes[window-1:], moving_avg_len, 'r-', linewidth=2, label=f'{window}-episode MA')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Episode Length (days)', fontsize=12)
        ax2.set_title('Episode Length During Training', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Training Loss
        ax3 = axes[1, 0]
        # Filter out None values
        valid_losses = [(i, loss) for i, loss in enumerate(losses) if loss is not None]
        if valid_losses:
            loss_steps, loss_values = zip(*valid_losses)
            ax3.plot(loss_steps, loss_values, 'purple', alpha=0.3, linewidth=0.5)
            # Moving average for loss
            loss_window = min(1000, len(loss_values) // 10)
            if loss_window > 0:
                loss_ma = np.convolve(loss_values, np.ones(loss_window)/loss_window, mode='valid')
                ax3.plot(loss_steps[loss_window-1:], loss_ma, 'r-', linewidth=2, label=f'{loss_window}-step MA')
            ax3.set_xlabel('Training Step', fontsize=12)
            ax3.set_ylabel('Loss (Huber)', fontsize=12)
            ax3.set_title('Training Loss', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No loss data available', ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Performance Summary
        ax4 = axes[1, 1]
        
        # Calculate metrics for last 100 episodes
        last_n = min(100, len(episode_rewards))
        recent_rewards = episode_rewards[-last_n:]
        
        metrics = ['Initial\nReward', 'Final\nReward', 'Best\nReward', 'Improvement']
        values = [
            np.mean(episode_rewards[:min(100, len(episode_rewards))]),
            np.mean(recent_rewards),
            np.max(episode_rewards),
            np.mean(recent_rewards) - np.mean(episode_rewards[:min(100, len(episode_rewards))])
        ]
        colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#1f77b4']
        
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Reward', fontsize=12)
        ax4.set_title('Training Performance Summary', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{save_dir}/training_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        if verbose > 0:
            print(f"\n✓ Training summary plot saved to: {plot_path}")
            print(f"\nTraining Summary:")
            print(f"  Initial mean reward (first {min(100, len(episode_rewards))} eps): {values[0]:.2f}")
            print(f"  Final mean reward (last {last_n} eps):   {values[1]:.2f}")
            print(f"  Best episode reward:    {values[2]:.2f}")
            print(f"  Improvement:            {values[3]:.2f}")
        
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
    hidden_dims=[128, 128],
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
    max_irrigation=15.0,
    n_actions=16
):
    """
    Train DQN agent on discrete irrigation environment.
    
    Parameters
    ----------
    total_timesteps : int
        Total number of timesteps to train (default: 200000)
    seed : int
        Random seed for reproducibility
    learning_rate : float
        Learning rate for optimizer (default: 1e-4)
    gamma : float
        Discount factor (default: 0.99)
    epsilon_start : float
        Initial exploration rate (default: 1.0)
    epsilon_end : float
        Final exploration rate (default: 0.05)
    epsilon_decay_steps : int
        Steps to decay epsilon (default: 50000)
    buffer_size : int
        Replay buffer capacity (default: 100000)
    batch_size : int
        Training batch size (default: 64)
    target_update_freq : int
        Target network update frequency (default: 1000)
    hidden_dims : list of int
        Hidden layer dimensions (default: [128, 128])
    save_dir : str
        Directory to save models
    log_dir : str
        Directory to save logs
    model_name : str, optional
        Name for saved model
    verbose : int
        Verbosity level
    max_et0, max_rain, et0_range, rain_range, max_soil_moisture, episode_length,
    threshold_bottom_soil_moisture, threshold_top_soil_moisture, max_irrigation :
        Environment configuration parameters (matching PPO)
    n_actions : int
        Number of discrete actions (default: 16)
    
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
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Print configuration
    if verbose > 0:
        print("="*80)
        print("DQN TRAINING - Discrete Action Space Irrigation")
        print("="*80)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Seed: {seed}")
        print(f"\nEnvironment: DiscretizedIrrigationEnv")
        print(f"  Episode length: {episode_length} days")
        print(f"  Max ET0: {max_et0} mm/day")
        print(f"  Rain range: {rain_range} mm/day")
        print(f"  Max soil moisture: {max_soil_moisture} mm")
        print(f"  Max irrigation: {max_irrigation} mm")
        print(f"  N actions: {n_actions} (0-{n_actions-1} mm)")
        print(f"\nDQN Configuration:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gamma: {gamma}")
        print(f"  Epsilon: {epsilon_start} -> {epsilon_end} (decay over {epsilon_decay_steps} steps)")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Target update freq: {target_update_freq}")
        print(f"  Hidden dims: {hidden_dims}")
        print("="*80)
    
    # Create environment
    env = make_discretized_env(
        max_et0=max_et0,
        max_rain=max_rain,
        et0_range=et0_range,
        rain_range=rain_range,
        max_soil_moisture=max_soil_moisture,
        episode_length=episode_length,
        threshold_bottom_soil_moisture=threshold_bottom_soil_moisture,
        threshold_top_soil_moisture=threshold_top_soil_moisture,
        max_irrigation=max_irrigation,
        n_actions=n_actions,
        seed=seed
    )
    
    # Create agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose > 0:
        print(f"\nUsing device: {device}")
    
    # State dimension: soil_moisture(1) + crop_stage_onehot(3) + rain(1) + et0(1) = 6
    STATE_DIM = 6
    
    agent = DQNAgent(
        state_dim=STATE_DIM,
        n_actions=n_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        hidden_dims=hidden_dims,
        device=device
    )
    
    # Training loop
    if verbose > 0:
        print("\nStarting training...")
        print("="*80)
    
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    for step in range(total_timesteps):
        # Select action
        action = agent.select_action(obs)
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)
        
        # Train agent
        loss = agent.train_step()
        losses.append(loss)
        
        # Update epsilon
        agent.update_epsilon()
        
        # Update episode stats
        episode_reward += reward
        episode_length += 1
        
        obs = next_obs
        
        # Episode end
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1
            
            # Log progress every 100 episodes
            if verbose > 0 and episode_count % 100 == 0:
                recent_rewards = episode_rewards[-100:]
                print(f"Episode {episode_count:4d} | Step {step+1:7d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg(100): {np.mean(recent_rewards):7.2f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
            
            # Reset environment
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
    
    if verbose > 0:
        print("="*80)
        print(f"Training complete!")
        print(f"Total episodes: {episode_count}")
        print(f"Total timesteps: {total_timesteps}")
    
    # Save final model
    if model_name is None:
        model_name = f"dqn_discrete_seed{seed}_steps{total_timesteps}.pt"
    
    save_path = f"{save_dir}/{model_name}"
    agent.save(save_path)
    
    if verbose > 0:
        print(f"Model saved to: {save_path}")
        print(f"Logs saved to: {log_dir}")
        print("="*80)
    
    # Generate training summary plots
    plot_training_summary(episode_rewards, episode_lengths, losses, save_dir, verbose=verbose)
    
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
                        help="Steps to decay epsilon")
    parser.add_argument("--buffer-size", type=int, default=100000,
                        help="Replay buffer size")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--target-update-freq", type=int, default=1000,
                        help="Target network update frequency")
    
    # Save/log parameters
    parser.add_argument("--save-dir", type=str, default="models/dqn_discrete",
                        help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs/dqn_discrete",
                        help="Directory to save logs")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Name for saved model")
    
    # Environment parameters
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
    parser.add_argument("--n-actions", type=int, default=16,
                        help="Number of discrete actions")
    
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
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        verbose=args.verbose,
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        episode_length=args.episode_length,
        max_soil_moisture=args.max_soil_moisture,
        max_irrigation=args.max_irrigation,
        n_actions=args.n_actions
    )
