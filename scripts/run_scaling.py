"""
Scaling Experiments:  Tabular Q-Learning vs DQN

Demonstrates how DQN scales better than tabular Q-learning
as the state space grows through finer discretization.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from src.envs.irrigation_env import IrrigationEnv
from src.agents.qtable import train_q_learning, discretize_state, get_state_space_size
from src.agents.dqn import train_dqn, evaluate_policy as evaluate_dqn
import torch


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_tabular_policy(Q, env, n_soil_bins, n_et0_bins, n_rain_bins, n_episodes=50):
    """
    Evaluate trained tabular Q-learning policy.
    
    Parameters
    ----------
    Q : np.ndarray
        Trained Q-table
    env : IrrigationEnv
        Environment instance
    n_soil_bins :  int
        Number of soil moisture bins
    n_et0_bins : int
        Number of ET0 bins
    n_rain_bins : int
        Number of rain bins
    n_episodes :  int
        Number of evaluation episodes
    
    Returns
    -------
    avg_return : float
        Average return over episodes
    returns : list
        List of episode returns
    """
    returns = []
    
    for _ in range(n_episodes):
        observation, _ = env.reset()
        done = False
        episode_return = 0
        
        while not done:
            # Get discrete state
            state = discretize_state(observation, n_soil_bins, n_et0_bins, n_rain_bins)
            
            # Greedy action selection
            action = np.argmax(Q[state])
            
            # Execute action
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
        
        returns.append(episode_return)
    
    return np.mean(returns), returns


# ============================================================================
# SCALING EXPERIMENT
# ============================================================================

def run_scaling_experiment(
    discretization_levels,
    n_train_episodes=1000,
    n_eval_episodes=50,
    seed=42
):
    """
    Run scaling experiment comparing tabular Q-learning and DQN.
    
    Parameters
    ----------
    discretization_levels : list of dict
        List of discretization configurations, each dict with keys:
        - 'n_soil_bins', 'n_et0_bins', 'n_rain_bins'
    n_train_episodes : int
        Number of training episodes
    n_eval_episodes :  int
        Number of evaluation episodes
    seed : int
        Random seed
    
    Returns
    -------
    results : dict
        Dictionary containing results for both agents
    """
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    env = IrrigationEnv(
        max_et0=8.0,
        max_rain=50.0,
        et0_range=(2.0, 8.0),
        rain_range=(0.0, 40.0),
        episode_length=90,
    )
    
    results = {
        'discretization_levels': discretization_levels,
        'state_space_sizes': [],
        'tabular_returns': [],
        'tabular_learning_curves': [],
        'dqn_returns': [],
        'dqn_learning_curves': []
    }
    
    print("="*80)
    print("SCALING EXPERIMENT:  Tabular Q-Learning vs DQN")
    print("="*80)
    print()
    
    # Train DQN once (doesn't depend on discretization)
    print("Training DQN (continuous state)...")
    print("-"*80)
    dqn_agent, dqn_learning_curve = train_dqn(
        env=env,
        n_episodes=n_train_episodes,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
        learning_rate=0.001,
        gamma=0.99,
        verbose=True
    )
    
    # Evaluate DQN
    print("\nEvaluating DQN...")
    dqn_avg_return, dqn_returns = evaluate_dqn(dqn_agent, env, n_episodes=n_eval_episodes)
    
    # Store DQN results (same for all discretization levels)
    results['dqn_learning_curves'] = dqn_learning_curve
    results['dqn_avg_return'] = dqn_avg_return
    
    print("\n" + "="*80)
    print()
    
    # Train tabular Q-learning for each discretization level
    for i, disc_level in enumerate(discretization_levels):
        n_soil = disc_level['n_soil_bins']
        n_et0 = disc_level['n_et0_bins']
        n_rain = disc_level['n_rain_bins']
        
        state_space_size = get_state_space_size(n_soil, n_et0_bins=n_et0, n_rain_bins=n_rain)
        results['state_space_sizes'].append(state_space_size)
        
        print(f"Discretization Level {i+1}/{len(discretization_levels)}")
        print(f"  Bins: soil={n_soil}, et0={n_et0}, rain={n_rain}")
        print(f"  State space size: {state_space_size}")
        print("-"*80)
        
        # Create new environment with same seed
        env_tabular = IrrigationEnv(
            max_et0=8.0,
            max_rain=50.0,
            et0_range=(2.0, 8.0),
            rain_range=(0.0, 40.0),
            episode_length=90,
        )
        
        # Train tabular Q-learning
        print("Training Tabular Q-Learning...")
        Q, _ = train_q_learning(
            env=env_tabular,
            n_episodes=n_train_episodes,
            alpha=0.1,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.995,
            n_soil_bins=n_soil,
            n_et0_bins=n_et0,
            n_rain_bins=n_rain,
            use_optimistic_init=True,
            verbose=False
        )
        
        # Evaluate tabular policy
        print("Evaluating Tabular Q-Learning...")
        tabular_avg_return, tabular_returns = evaluate_tabular_policy(
            Q, env_tabular, n_soil, n_et0, n_rain, n_episodes=n_eval_episodes
        )
        
        print(f"  Tabular Avg Return: {tabular_avg_return:.2f}")
        print(f"  DQN Avg Return: {dqn_avg_return:.2f}")
        
        results['tabular_returns'].append(tabular_avg_return)
        
        print()
    
    print("="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_scaling_results(results, save_path=None):
    """
    Plot scaling experiment results.
    
    Parameters
    ----------
    results : dict
        Results from run_scaling_experiment()
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Performance vs State Space Size
    ax1 = axes[0]
    state_space_sizes = results['state_space_sizes']
    tabular_returns = results['tabular_returns']
    dqn_avg_return = results['dqn_avg_return']
    
    ax1.plot(state_space_sizes, tabular_returns, 'o-', label='Tabular Q-Learning', linewidth=2, markersize=8)
    ax1.axhline(y=dqn_avg_return, color='red', linestyle='--', label='DQN (continuous state)', linewidth=2)
    ax1.set_xlabel('State Space Size', fontsize=12)
    ax1.set_ylabel('Average Return (50 eval episodes)', fontsize=12)
    ax1.set_title('Scaling Comparison: Performance vs State Space Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning Curves
    ax2 = axes[1]
    
    # DQN learning curve
    dqn_curve = results['dqn_learning_curves']
    window = 50
    dqn_smoothed = np.convolve(dqn_curve, np. ones(window)/window, mode='valid')
    ax2.plot(range(len(dqn_smoothed)), dqn_smoothed, label='DQN', linewidth=2, color='red')
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Return (smoothed)', fontsize=12)
    ax2.set_title('Learning Curve: DQN', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Define discretization levels (coarse → fine)
    discretization_levels = [
        {'n_soil_bins': 6, 'n_et0_bins': 4, 'n_rain_bins': 3},    # 216 states
        {'n_soil_bins': 12, 'n_et0_bins': 8, 'n_rain_bins':  6},   # 1,728 states
        {'n_soil_bins': 18, 'n_et0_bins': 12, 'n_rain_bins': 9},  # 5,832 states
    ]
    
    # Run experiment
    results = run_scaling_experiment(
        discretization_levels=discretization_levels,
        n_train_episodes=1000,
        n_eval_episodes=50,
        seed=42
    )
    
    # Plot results
    plot_scaling_results(results, save_path='scaling_results.png')
    
    # Print summary
    print("\nSUMMARY:")
    print("-"*80)
    print(f"{'Discretization':<30} {'State Space':<15} {'Tabular Return':<20}")
    print("-"*80)
    for i, disc in enumerate(results['discretization_levels']):
        disc_str = f"soil={disc['n_soil_bins']}, et0={disc['n_et0_bins']}, rain={disc['n_rain_bins']}"
        state_size = results['state_space_sizes'][i]
        tabular_return = results['tabular_returns'][i]
        print(f"{disc_str:<30} {state_size:<15} {tabular_return:<20.2f}")
    print("-"*80)
    print(f"{'DQN (continuous)':<30} {'N/A':<15} {results['dqn_avg_return']: <20.2f}")
    print("-"*80)