"""
PPO Training Script for Continuous Action Space Irrigation

Trains PPO agent on IrrigationEnvContinuous with continuous irrigation amounts [0, max_irrigation] mm.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from ppo_env import make_irrigation_env, make_vec_env


def plot_training_results(log_dir, save_dir, verbose=1):
    """
    Create summary plots of training progress.
    
    Parameters
    ----------
    log_dir : str
        Directory containing training logs
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
        # Read evaluation results
        eval_npz = f"{log_dir}/eval/evaluations.npz"
        if os.path.exists(eval_npz):
            eval_data = np.load(eval_npz)
            timesteps = eval_data['timesteps']
            results = eval_data['results']
            ep_lengths = eval_data['ep_lengths']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('PPO Training Summary - Continuous Irrigation', fontsize=16, fontweight='bold')
            
            # Plot 1: Mean Episode Reward over Time
            ax1 = axes[0, 0]
            mean_rewards = np.mean(results, axis=1)
            std_rewards = np.std(results, axis=1)
            ax1.plot(timesteps, mean_rewards, 'b-', linewidth=2, label='Mean Reward')
            ax1.fill_between(timesteps, 
                            mean_rewards - std_rewards, 
                            mean_rewards + std_rewards, 
                            alpha=0.3, color='b', label='Std Dev')
            ax1.set_xlabel('Timesteps', fontsize=12)
            ax1.set_ylabel('Episode Reward', fontsize=12)
            ax1.set_title('Episode Rewards During Training', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Episode Length over Time
            ax2 = axes[0, 1]
            mean_lengths = np.mean(ep_lengths, axis=1)
            std_lengths = np.std(ep_lengths, axis=1)
            ax2.plot(timesteps, mean_lengths, 'g-', linewidth=2, label='Mean Length')
            ax2.fill_between(timesteps,
                            mean_lengths - std_lengths,
                            mean_lengths + std_lengths,
                            alpha=0.3, color='g', label='Std Dev')
            ax2.set_xlabel('Timesteps', fontsize=12)
            ax2.set_ylabel('Episode Length (days)', fontsize=12)
            ax2.set_title('Episode Length During Training', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Reward Distribution (First vs Last Evaluation)
            ax3 = axes[1, 0]
            first_eval = results[0]
            last_eval = results[-1]
            ax3.hist(first_eval, bins=20, alpha=0.5, label=f'First Eval (t={timesteps[0]})', color='orange')
            ax3.hist(last_eval, bins=20, alpha=0.5, label=f'Last Eval (t={timesteps[-1]})', color='purple')
            ax3.set_xlabel('Episode Reward', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Reward Distribution: First vs Last Evaluation', fontsize=13, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Performance Improvement Summary
            ax4 = axes[1, 1]
            
            # Calculate improvement metrics
            initial_mean = mean_rewards[0]
            final_mean = mean_rewards[-1]
            best_mean = np.max(mean_rewards)
            improvement = final_mean - initial_mean
            best_improvement = best_mean - initial_mean
            
            metrics = ['Initial\nReward', 'Final\nReward', 'Best\nReward', 'Improvement\n(Final-Initial)']
            values = [initial_mean, final_mean, best_mean, improvement]
            colors = ['#ff7f0e', '#2ca02c', '#9467bd', '#1f77b4']
            
            bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax4.set_ylabel('Reward', fontsize=12)
            ax4.set_title('Training Performance Summary', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add value labels on bars
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
                print(f"  Initial mean reward: {initial_mean:.2f}")
                print(f"  Final mean reward:   {final_mean:.2f}")
                print(f"  Best mean reward:    {best_mean:.2f}")
                print(f"  Improvement:         {improvement:.2f} ({improvement/abs(initial_mean)*100:.1f}%)")
            
            plt.close()
            
        else:
            if verbose > 0:
                print(f"⚠ Evaluation file not found: {eval_npz}")
                print("  Plots will not be generated.")
    
    except Exception as e:
        if verbose > 0:
            print(f"⚠ Error creating plots: {e}")


def train_ppo_continuous(
    total_timesteps=200000,
    seed=42,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_envs=4,
    eval_freq=10000,
    n_eval_episodes=50,
    save_dir="models/ppo_continuous",
    log_dir="logs/ppo_continuous",
    model_name=None,
    load_model=None,
    verbose=1,
    # Environment parameters
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
    Train PPO agent on continuous irrigation environment.
    
    Parameters
    ----------
    total_timesteps : int
        Total number of timesteps to train
    seed : int
        Random seed for reproducibility
    learning_rate : float
        Learning rate for optimizer
    n_steps : int
        Number of steps to run for each environment per update
    batch_size : int
        Minibatch size
    n_epochs : int
        Number of epochs when optimizing the surrogate loss
    gamma : float
        Discount factor
    gae_lambda : float
        Factor for trade-off of bias vs variance for GAE
    clip_range : float
        Clipping parameter for PPO
    ent_coef : float
        Entropy coefficient for the loss calculation (encourages exploration)
    vf_coef : float
        Value function coefficient for the loss calculation
    max_grad_norm : float
        Maximum value for gradient clipping
    n_envs : int
        Number of parallel environments
    eval_freq : int
        Evaluate the agent every eval_freq timesteps
    n_eval_episodes : int
        Number of episodes for evaluation
    save_dir : str
        Directory to save trained model
    log_dir : str
        Directory to save training logs
    model_name : str, optional
        Name for saved model (if None, auto-generates based on seed and timesteps)
    load_model : str, optional
        Path to existing model to load and continue training from.
        If provided, loads the model instead of creating a new one.
    verbose : int
        Verbosity level (0=none, 1=info, 2=debug)
    max_et0, max_rain, et0_range, rain_range, max_soil_moisture, episode_length,
    threshold_bottom_soil_moisture, threshold_top_soil_moisture :
        Environment configuration parameters
    max_irrigation : float
        Maximum irrigation amount for continuous action space (mm)
    
    Returns
    -------
    model : PPO
        Trained PPO model
    save_path : str
        Path where model was saved
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/best_model", exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/train", exist_ok=True)
    os.makedirs(f"{log_dir}/eval", exist_ok=True)
    
    # Set seeds
    np.random.seed(seed)
    
    # Environment kwargs
    env_kwargs = {
        "max_et0": max_et0,
        "max_rain": max_rain,
        "et0_range": et0_range,
        "rain_range": rain_range,
        "max_soil_moisture": max_soil_moisture,
        "episode_length": episode_length,
        "threshold_bottom_soil_moisture": threshold_bottom_soil_moisture,
        "threshold_top_soil_moisture": threshold_top_soil_moisture,
        "continuous": True,  # Enable continuous action space
        "max_irrigation": max_irrigation
    }
    
    # Print configuration
    if verbose > 0:
        print("="*80)
        print("PPO TRAINING - Continuous Action Space Irrigation")
        print("="*80)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Seed: {seed}")
        print(f"Number of parallel environments: {n_envs}")
        print(f"\nEnvironment: IrrigationEnvContinuous")
        print(f"  Episode length: {episode_length} days")
        print(f"  Max ET0: {max_et0} mm/day")
        print(f"  Rain range: {rain_range} mm/day")
        print(f"  Max soil moisture: {max_soil_moisture} mm")
        print(f"  Max irrigation: {max_irrigation} mm (continuous)")
        print(f"\nPPO Configuration:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Entropy coefficient: {ent_coef}")
        print(f"  N steps: {n_steps}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gamma: {gamma}")
        print("="*80)
    
    # Create training environment (vectorized)
    if n_envs > 1:
        env = make_vec_env(
            n_envs=n_envs,
            seed=seed,
            monitor_dir=f"{log_dir}/train",
            **env_kwargs
        )
    else:
        env = make_irrigation_env(
            seed=seed,
            monitor_filename=f"{log_dir}/train/env_0",
            **env_kwargs
        )
    
    # Create evaluation environment
    eval_env = make_irrigation_env(
        seed=seed + 1000,
        monitor_filename=f"{log_dir}/eval/monitor",
        **env_kwargs
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model",
        log_path=f"{log_dir}/eval",
        eval_freq=max(eval_freq // n_envs, 1),  # Adjust for vectorized envs
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),  # Save every 50k steps
        save_path=f"{save_dir}/checkpoints",
        name_prefix="ppo_continuous",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=verbose
    )
    
    # Initialize or load PPO model
    if load_model is not None:
        if verbose > 0:
            print(f"\nLoading existing model from: {load_model}")
        
        if not os.path.exists(load_model):
            raise FileNotFoundError(f"Model file not found: {load_model}")
        
        model = PPO.load(
            load_model,
            env=env,
            tensorboard_log=log_dir,
            verbose=verbose
        )
        
        if verbose > 0:
            print("✓ Model loaded successfully")
            print("  Continuing training from existing model...")
    else:
        if verbose > 0:
            print("\nInitializing new PPO model with MultiInputPolicy...")
        
        model = PPO(
            "MultiInputPolicy",  # Required for Dict observation space
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,  # Important for continuous action exploration
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            seed=seed,
            verbose=verbose,
            tensorboard_log=log_dir
        )
    
    # Train model
    if verbose > 0:
        print("\nStarting training...")
        print("="*80)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=(verbose > 0)
    )
    
    # Save final model
    if model_name is None:
        model_name = f"ppo_continuous_seed{seed}_steps{total_timesteps}"
    
    save_path = f"{save_dir}/{model_name}"
    model.save(save_path)
    
    if verbose > 0:
        print("="*80)
        print(f"Training complete!")
        print(f"Model saved to: {save_path}.zip")
        print(f"Best model saved to: {save_dir}/best_model/best_model.zip")
        print(f"Logs saved to: {log_dir}")
        print(f"\nTo monitor training:")
        print(f"  tensorboard --logdir {log_dir}")
        print("="*80)
    
    # Generate training summary plots
    plot_training_results(log_dir, save_dir, verbose=verbose)
    
    return model, save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO agent on continuous irrigation scheduling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument("--timesteps", type=int, default=400000,
                        help="Total timesteps for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per environment per update")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of epochs for optimization")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (for exploration)")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel environments")
    
    # Evaluation parameters
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency (timesteps)")
    parser.add_argument("--n-eval-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    
    # Save/log parameters
    parser.add_argument("--save-dir", type=str, default="models/ppo_continuous",
                        help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs/ppo_continuous",
                        help="Directory to save logs")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Name for saved model")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to existing model to load and continue training (e.g., models/ppo_continuous/best_model/best_model.zip)")
    
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
    
    # Verbosity
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=none, 1=info, 2=debug)")
    
    args = parser.parse_args()
    
    # Interactive prompt for loading existing model (if not specified in args)
    if args.load_model is None and args.verbose > 0:
        response = input("\nLoad existing model? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            model_path = input("Enter model path (e.g., models/ppo_continuous/best_model/best_model.zip): ").strip()
            if model_path and os.path.exists(model_path):
                args.load_model = model_path
            elif model_path:
                print(f"⚠ Model file not found: {model_path}")
                print("Continuing with new model...")
    
    # Train model
    model, save_path = train_ppo_continuous(
        total_timesteps=args.timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        load_model=args.load_model,
        verbose=args.verbose,
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        episode_length=args.episode_length,
        max_soil_moisture=args.max_soil_moisture,
        max_irrigation=args.max_irrigation
    )
