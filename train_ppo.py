"""
PPO Training Script for Irrigation Scheduling

Trains a PPO agent using stable-baselines3 on the IrrigationEnv.
"""

import argparse
import os
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from ppo_env import make_irrigation_env, make_vec_env


def train_ppo(
    total_timesteps=100000,
    seed=42,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    n_envs=1,
    eval_freq=10000,
    n_eval_episodes=50,
    save_dir="models/ppo",
    log_dir="logs/ppo",
    model_name=None,
    verbose=1,
    # Environment parameters
    max_et0=8.0,
    max_rain=50.0,
    et0_range=(2.0, 8.0),
    rain_range=(0.0, 0.8),
    max_soil_moisture=320.0,
    episode_length=90,
    threshold_bottom_soil_moisture=0.4,
    threshold_top_soil_moisture=0.7
):
    """
    Train PPO agent on irrigation environment.
    
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
        Entropy coefficient for the loss calculation
    vf_coef :  float
        Value function coefficient for the loss calculation
    max_grad_norm : float
        Maximum value for gradient clipping
    n_envs : int
        Number of parallel environments
    eval_freq : int
        Evaluate the agent every eval_freq timesteps
    n_eval_episodes : int
        Number of episodes for evaluation
    save_dir :  str
        Directory to save trained model
    log_dir : str
        Directory to save training logs
    model_name : str, optional
        Name for saved model (if None, auto-generates based on seed and timesteps)
    verbose : int
        Verbosity level (0=none, 1=info, 2=debug)
    max_et0, max_rain, et0_range, rain_range, max_soil_moisture, episode_length,
    threshold_bottom_soil_moisture, threshold_top_soil_moisture :  
        Environment configuration parameters
    
    Returns
    -------
    model : PPO
        Trained PPO model
    save_path : str
        Path where model was saved
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Set seeds
    np.random.seed(seed)
    
    # Environment kwargs
    env_kwargs = {
        "max_et0": max_et0,
        "max_rain": max_rain,
        "et0_range": et0_range,
        "rain_range": rain_range,
        "max_soil_moisture":  max_soil_moisture,
        "episode_length": episode_length,
        "threshold_bottom_soil_moisture": threshold_bottom_soil_moisture,
        "threshold_top_soil_moisture": threshold_top_soil_moisture,
    }
    
    # Create training environment
    if verbose > 0:
        print("="*80)
        print("PPO TRAINING - Irrigation Scheduling")
        print("="*80)
        print(f"Total timesteps: {total_timesteps}")
        print(f"Seed: {seed}")
        print(f"Number of parallel environments: {n_envs}")
        print(f"Environment:  IrrigationEnv")
        print(f"  Episode length: {episode_length} days")
        print(f"  Max ET0: {max_et0} mm/day")
        print(f"  Rain range: {rain_range} mm/day")
        print(f"  Max soil moisture: {max_soil_moisture} mm")
        print("="*80)
    
    if n_envs > 1:
        env = make_vec_env(n_envs=n_envs, seed=seed, **env_kwargs)
    else:
        env = make_irrigation_env(seed=seed, **env_kwargs)
    
    # Create evaluation environment
    eval_env = make_irrigation_env(seed=seed + 1000, **env_kwargs)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best",
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq, 10000),
        save_path=f"{save_dir}/checkpoints",
        name_prefix="ppo_irrigation",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=verbose
    )
    
    # Initialize PPO model
    if verbose > 0:
        print("\nInitializing PPO model...")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  N steps: {n_steps}")
        print(f"  N epochs: {n_epochs}")
        print(f"  Gamma: {gamma}")
        print(f"  GAE lambda: {gae_lambda}")
        print(f"  Clip range: {clip_range}")
    
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
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
        model_name = f"ppo_irrigation_seed{seed}_steps{total_timesteps}"
    
    save_path = f"{save_dir}/{model_name}"
    model.save(save_path)
    
    if verbose > 0:
        print("="*80)
        print(f"Training complete!")
        print(f"Model saved to: {save_path}. zip")
        print(f"Logs saved to: {log_dir}")
        print("="*80)
    
    return model, save_path


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Train PPO agent on irrigation scheduling")
    
    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=100000,
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
    parser.add_argument("--n-envs", type=int, default=1,
                        help="Number of parallel environments")
    
    # Evaluation parameters
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency (timesteps)")
    parser.add_argument("--n-eval-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    
    # Save/log parameters
    parser.add_argument("--save-dir", type=str, default="models/ppo",
                        help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs/ppo",
                        help="Directory to save logs")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Name for saved model")
    
    # Environment parameters
    parser.add_argument("--max-et0", type=float, default=8.0,
                        help="Maximum ET0 (mm/day)")
    parser.add_argument("--max-rain", type=float, default=50.0,
                        help="Maximum rainfall (mm/day)")
    parser.add_argument("--episode-length", type=int, default=90,
                        help="Episode length (days)")
    parser.add_argument("--max-soil-moisture", type=float, default=320.0,
                        help="Maximum soil moisture capacity (mm)")
    
    # Verbosity
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=none, 1=info, 2=debug)")
    
    args = parser.parse_args()
    
    # Train model
    model, save_path = train_ppo(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        n_envs=args.n_envs,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        model_name=args.model_name,
        verbose=args.verbose,
        max_et0=args.max_et0,
        max_rain=args.max_rain,
        episode_length=args.episode_length,
        max_soil_moisture=args.max_soil_moisture
    )