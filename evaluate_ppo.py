"""
PPO Evaluation Script for Irrigation Scheduling

Evaluates trained PPO agents using the same format as DQN evaluation.
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from ppo_env import make_irrigation_env
from ppo_rollout import run_ppo_rollout


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
        print("="*80)