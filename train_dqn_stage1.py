"""
Train DQN specifically for crop stage 1 (Flowering).

This script creates a specialized training environment that focuses on the 
flowering stage, where water stress is most critical.
"""

from irrigation_env import IrrigationEnv
from irr_dqn import train_dqn, run_policy_rollout, evaluate_policy
from pathlib import Path
import numpy as np


class Stage1FocusedEnv(IrrigationEnv):
    """
    Environment wrapper that focuses training on crop stage 1 (Flowering).
    
    Resets to day 31 (start of flowering) with varied initial conditions.
    """
    
    def reset(self, seed=None, options=None):
        """Reset to beginning of flowering stage."""
        obs, info = super().reset(seed=seed, options=options)
        
        # Force to start at flowering stage (day 31)
        self.current_step = 30  # Will become 31 on first step
        self.crop_stage = 1  # Flowering
        
        # Randomize initial soil moisture in reasonable range
        self.soil_moisture = np.random.uniform(0.3, 0.7)
        
        # Update observation
        obs = self._get_obs()
        self.prev_soil_moisture = self.soil_moisture
        
        return obs, info


def main():
    """Train DQN focused on flowering stage."""
    
    print("="*80)
    print("TRAINING DQN FOR CROP STAGE 1 (FLOWERING)")
    print("="*80)
    print()
    print("This training focuses on the flowering stage where water")
    print("stress has the most critical impact on crop yield.")
    print()
    
    # Create specialized environment
    env = Stage1FocusedEnv(
        max_et0=50.0,
        max_rain=20.0,
        et0_range=(2.0, 50.0),
        rain_range=(0.0, 20.0),
        episode_length=60,  # 60 days: covers flowering + maturity
        reset_soil_moisture_range=(0.3, 0.7)  # Start with varied moisture
    )
    
    print(f"Environment configuration:")
    print(f"  Starting stage: 1 (Flowering)")
    print(f"  Episode length: 60 days")
    print(f"  Initial SM range: 0.3 - 0.7")
    print(f"  State dimension: 4")
    print(f"  Action space: {env.action_space.n}")
    print()
    
    # Check if there's an existing general model to continue from
    save_dir = "models/dqn_stage1"
    load_checkpoint = None
    
    print("="*80)
    print("TRAINING OPTIONS")
    print("="*80)
    
    general_model = Path("models/dqn/dqn_final.pt")
    if general_model.exists():
        print(f"\nFound existing general model: {general_model}")
        print("\nChoose training approach:")
        print("  1. Fine-tune from general model (RECOMMENDED - faster, better results)")
        print("     → Starts with already learned knowledge, adapts to stage 1")
        print("     → Original model remains safe in models/dqn/")
        print("  2. Train from scratch")
        print("     → Starts from random weights, learns stage 1 only")
        print()
        response = input("Enter choice (1 or 2): ").strip()
        
        if response == '1':
            load_checkpoint = str(general_model)
            print("\n✓ Will fine-tune from general model")
            print(f"✓ New model will be saved to: {save_dir}/")
            print(f"✓ Original model safe at: models/dqn/\n")
        else:
            print("\n✓ Starting fresh training from scratch")
            print(f"✓ Model will be saved to: {save_dir}/\n")
    else:
        print("\nNo existing general model found.")
        print(f"Starting fresh training for stage 1.")
        print(f"Model will be saved to: {save_dir}/\n")
    
    # Train agent focused on stage 1
    print("Starting training...")
    print("-"*80)
    
    agent, episode_returns = train_dqn(
        env=env,
        n_episodes=5000,
        epsilon_start=1.0 if load_checkpoint is None else 0.3,  # Less exploration if continuing
        epsilon_end=0.1,
        epsilon_decay=0.999,
        learning_rate=0.001,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        hidden_size=64,
        verbose=True,
        save_dir=save_dir,
        checkpoint_freq=500,
        load_checkpoint=load_checkpoint
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - EVALUATING ON STAGE 1")
    print("="*80)
    
    # Evaluate on stage 1 focused episodes
    avg_return, returns = evaluate_policy(agent, env, n_episodes=50, verbose=True)
    
    # Test on full season starting from beginning
    print("\n" + "="*80)
    print("TESTING ON FULL SEASON")
    print("="*80)
    
    full_env = IrrigationEnv(
        max_et0=50.0,
        max_rain=20.0,
        et0_range=(2.0, 50.0),
        rain_range=(0.0, 20.0),
        episode_length=90,
    )
    
    print("\nFull season rollout (seed=42):")
    rollout_data = run_policy_rollout(agent, full_env, seed=42, verbose=True)
    
    # Analyze stage 1 performance
    print("\n" + "="*80)
    print("STAGE 1 PERFORMANCE ANALYSIS")
    print("="*80)
    
    stage1_data = [d for d in rollout_data if d['crop_stage'] == 1]
    if stage1_data:
        stage1_sm = [d['soil_moisture'] for d in stage1_data]
        stage1_irr = [d['irrigation'] for d in stage1_data]
        
        print(f"\nFlowering stage (stage 1) - Days 31-60:")
        print(f"  Average soil moisture: {np.mean(stage1_sm):.3f}")
        print(f"  Min soil moisture: {np.min(stage1_sm):.3f}")
        print(f"  Max soil moisture: {np.max(stage1_sm):.3f}")
        print(f"  Total irrigation: {np.sum(stage1_irr):.1f} mm")
        print(f"  Days in optimal range (0.4-0.7): {sum(1 for sm in stage1_sm if 0.4 <= sm <= 0.7)}/{len(stage1_sm)}")
        print(f"  Days below 0.4: {sum(1 for sm in stage1_sm if sm < 0.4)}")
        print(f"  Days above 0.7: {sum(1 for sm in stage1_sm if sm > 0.7)}")
    
    # Compare with other stages
    stage0_data = [d for d in rollout_data if d['crop_stage'] == 0]
    stage2_data = [d for d in rollout_data if d['crop_stage'] == 2]
    
    print(f"\nComparison across all stages:")
    for stage, data, name in [(0, stage0_data, "Emergence"), 
                               (1, stage1_data, "Flowering"), 
                               (2, stage2_data, "Maturity")]:
        if data:
            sm = [d['soil_moisture'] for d in data]
            irr = [d['irrigation'] for d in data]
            print(f"  {name:12s}: Avg SM={np.mean(sm):.3f}, Total Irr={np.sum(irr):5.0f} mm")
    
    print("\n" + "="*80)
    print(f"Stage 1 focused model saved to: {save_dir}/dqn_final.pt")
    print("="*80)


if __name__ == "__main__":
    main()
