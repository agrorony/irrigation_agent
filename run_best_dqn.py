"""
Run policy rollout using the latest saved DQN model.

This script:
1. Finds the most recent checkpoint in models/dqn/
2. Loads the DQN agent
3. Runs a detailed rollout showing the agent's behavior
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from irrigation_agent.irrigation_env import IrrigationEnv
from irr_dqn import DQNAgent, run_policy_rollout, evaluate_policy


def find_latest_checkpoint(model_dir="models/dqn"):
    """
    Find the latest checkpoint or final model in the specified directory.
    
    Parameters
    ----------
    model_dir : str
        Directory containing model checkpoints
    
    Returns
    -------
    latest_path : Path or None
        Path to the latest checkpoint, or None if no models found
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        return None
    
    # Look for checkpoint files
    checkpoints = list(model_path.glob("checkpoint_ep*.pt"))
    
    # Also check for final model
    final_model = model_path / "dqn_final.pt"
    if final_model.exists():
        checkpoints.append(final_model)
    
    if not checkpoints:
        return None
    
    # Sort by modification time and get the most recent
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    
    return latest


def plot_rollout(rollout_data, save_path=None):
    """
    Plot soil moisture and actions over time from rollout data.
    
    Parameters
    ----------
    rollout_data : list of dict
        Rollout data from run_policy_rollout
    save_path : str or Path, optional
        Path to save the figure. If None, displays interactively.
    """
    # Extract data
    days = [d['day'] for d in rollout_data]
    soil_moisture = [d['soil_moisture'] for d in rollout_data]
    actions = [d['action'] for d in rollout_data]
    irrigation = [d['irrigation'] for d in rollout_data]
    rain = [d['rain'] for d in rollout_data]
    et0 = [d['et0'] for d in rollout_data]
    crop_stage = [d['crop_stage'] for d in rollout_data]
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Soil Moisture
    ax1.plot(days, soil_moisture, 'b-', linewidth=2, label='Soil Moisture')
    ax1.axhline(y=0.4, color='r', linestyle='--', alpha=0.5, label='Lower Threshold')
    ax1.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Upper Threshold')
    ax1.fill_between(days, 0.4, 0.7, alpha=0.1, color='green', label='Optimal Range')
    ax1.set_ylabel('Soil Moisture (fraction)', fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_title('DQN Policy Rollout - Soil Moisture and Irrigation Actions', fontsize=14, fontweight='bold')
    
    # Plot 2: Actions and Irrigation
    # Color code the actions
    colors = ['gray', 'lightblue', 'blue']
    action_names = ['No Irrigation', 'Light (5mm)', 'Heavy (15mm)']
    
    for action_val in [0, 1, 2]:
        action_days = [d for d, a in zip(days, actions) if a == action_val]
        action_irr = [i for a, i in zip(actions, irrigation) if a == action_val]
        if action_days:
            ax2.bar(action_days, action_irr, color=colors[action_val], 
                   label=action_names[action_val], alpha=0.8, width=0.8)
    
    ax2.set_ylabel('Irrigation (mm)', fontsize=12)
    ax2.set_ylim([0, max(irrigation) * 1.1 if max(irrigation) > 0 else 20])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right')
    
    # Plot 3: Climate conditions (ET0 and Rain)
    ax3_twin = ax3.twinx()
    ax3.bar(days, rain, color='skyblue', alpha=0.6, label='Rain', width=0.8)
    ax3_twin.plot(days, et0, 'r-', linewidth=2, label='ET0', marker='.')
    
    ax3.set_xlabel('Day', fontsize=12)
    ax3.set_ylabel('Rain (mm/day)', fontsize=12, color='skyblue')
    ax3_twin.set_ylabel('ET0 (mm/day)', fontsize=12, color='red')
    ax3.tick_params(axis='y', labelcolor='skyblue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add crop stage markers
    stage_changes = [0]
    for i in range(1, len(crop_stage)):
        if crop_stage[i] != crop_stage[i-1]:
            stage_changes.append(i)
    stage_changes.append(len(crop_stage))
    
    stage_names = ['Emergence', 'Flowering', 'Maturity']
    for i in range(len(stage_changes)-1):
        start_day = days[stage_changes[i]]
        end_day = days[stage_changes[i+1]-1] if stage_changes[i+1] < len(days) else days[-1]
        mid_day = (start_day + end_day) / 2
        stage = crop_stage[stage_changes[i]]
        ax1.axvspan(start_day-0.5, end_day+0.5, alpha=0.05, color=['orange', 'green', 'brown'][stage])
        ax1.text(mid_day, 1.02, stage_names[stage], ha='center', va='bottom', 
                fontsize=9, style='italic', color=['orange', 'green', 'brown'][stage])
    
    # Combine legends for plot 3
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    else:
        plt.show()


def main():
    """Main execution: load latest model and run rollout."""
    
    print("="*80)
    print("DQN POLICY ROLLOUT - Latest Saved Model")
    print("="*80)
    print()
    
    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint("models/dqn")
    
    if checkpoint_path is None:
        print("ERROR: No saved models found in models/dqn/")
        print("Please train a model first by running irr_dqn.py")
        return
    
    print(f"Found model: {checkpoint_path}")
    print(f"Last modified: {checkpoint_path.stat().st_mtime}")
    print()
    
    # Create environment (must match training configuration)
    env = IrrigationEnv(
        max_et0=50.0,
        max_rain=20.0,
        et0_range=(2.0, 50.0),
        rain_range=(0.0, 20.0),
        episode_length=90,
    )
    
    print(f"Environment created:")
    print(f"  State dimension: 4")
    print(f"  Action space: {env.action_space.n}")
    print()
    
    # Initialize agent with same hyperparameters as training
    state_dim = 4
    n_actions = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        learning_rate=0.001,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        hidden_size=64
    )
    
    # Load checkpoint
    print("Loading model...")
    agent.load(checkpoint_path)
    print()
    
    # Run evaluation on multiple episodes
    print("="*80)
    print("EVALUATING AGENT PERFORMANCE")
    print("="*80)
    avg_return, returns = evaluate_policy(agent, env, n_episodes=50, verbose=True)
    
    # Run detailed rollout
    print("\n" + "="*80)
    print("DETAILED POLICY ROLLOUT (seed=42)")
    print("="*80)
    rollout_data = run_policy_rollout(agent, env, seed=42, verbose=True)
    
    # Plot the rollout
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    plot_rollout(rollout_data, save_path="models/dqn/rollout_visualization.png")
    
    # Additional rollouts with different seeds
    print("\n" + "="*80)
    print("ADDITIONAL ROLLOUTS")
    print("="*80)
    
    for seed in [123, 456, 789]:
        print(f"\nRollout with seed={seed}:")
        print("-"*80)
        rollout = run_policy_rollout(agent, env, seed=seed, verbose=False)
        total_reward = sum(d['reward'] for d in rollout)
        total_irrigation = sum(d['irrigation'] for d in rollout)
        avg_sm = sum(d['soil_moisture'] for d in rollout) / len(rollout)
        
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Total irrigation: {total_irrigation:.1f} mm")
        print(f"  Average soil moisture: {avg_sm:.3f}")


if __name__ == "__main__":
    main()
