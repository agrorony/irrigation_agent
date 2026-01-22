# DQN Discrete Agent - Usage Guide

## Overview

This implementation provides a vanilla DQN (Deep Q-Network) agent for the irrigation scheduling task with discretized actions. It uses the same environment dynamics and reward function as the PPO continuous implementation, enabling fair comparison between the two approaches.

## Key Features

- **Discretized Action Space**: 16 discrete actions (0-15 mm irrigation)
- **Same Environment**: Uses `IrrigationEnvContinuous` with discretization wrapper
- **Vanilla DQN**: Simple implementation without Double DQN or other enhancements
- **PyTorch Implementation**: From scratch, not using stable-baselines3
- **Matching Evaluation**: Same metrics as PPO for fair comparison

## Files

1. **`dqn_wrapper.py`**: Discretization wrapper for continuous environment
2. **`dqn_agent.py`**: DQN agent implementation with replay buffer
3. **`train_dqn_discrete.py`**: Training script with CLI
4. **`evaluate_dqn_discrete.py`**: Evaluation script with metrics and plots

## Quick Start

### Training

```bash
# Train with default parameters (200k timesteps)
python train_dqn_discrete.py --timesteps 200000 --seed 42

# Train with custom parameters
python train_dqn_discrete.py \
    --timesteps 400000 \
    --learning-rate 1e-4 \
    --epsilon-decay-steps 100000 \
    --batch-size 64 \
    --seed 42
```

### Evaluation

```bash
# Evaluate trained model
python evaluate_dqn_discrete.py \
    --model-path models/dqn_discrete/dqn_discrete_seed42_steps200000.pt \
    --n-episodes 50 \
    --seed 42

# Evaluate with custom environment parameters
python evaluate_dqn_discrete.py \
    --model-path models/dqn_discrete/dqn_discrete_seed42_steps200000.pt \
    --n-episodes 100 \
    --rain-range 0.0 10.0 \
    --max-irrigation 20.0
```

## Default Hyperparameters

### DQN Agent
- Learning rate: `1e-4`
- Gamma (discount): `0.99`
- Epsilon decay: `1.0` → `0.05` over `50,000` steps
- Replay buffer: `100,000` transitions
- Batch size: `64`
- Target network update: every `1,000` steps
- Hidden layers: `[128, 128]`

### Environment (matching PPO)
- Episode length: `90` days
- Max ET₀: `15.0` mm/day
- Rain range: `(0.0, 5.0)` mm/day
- ET₀ range: `(2.0, 15.0)` mm/day
- Max soil moisture: `100.0` mm
- Optimal range: `[0.4, 0.7]`
- Max irrigation: `15.0` mm
- N actions: `16` (0-15 mm discrete)

## Evaluation Metrics

The evaluation script provides the same metrics as PPO evaluation:

1. **Episode Rewards**: mean, std, min, max
2. **Irrigation**: mean, std (mm per episode and per step)
3. **Action Distribution**: zero, light, medium, heavy percentages
4. **Soil Moisture Performance**:
   - `optimal_pct`: percentage of time in [0.4, 0.7] range
   - `water_stress_events`: count of timesteps with SM < 0.4

## Outputs

### Training
- Model checkpoints: `models/dqn_discrete/`
- Training summary plot: `models/dqn_discrete/training_summary.png`

### Evaluation
- Performance plots: `logs/dqn_discrete/evaluation_performance.png`
- Episode rollout: `logs/dqn_discrete/episode_rollout.png`
- CSV logs: `logs/dqn_discrete/evaluation_logs.csv` (with `--save-logs`)

## Example: Complete Workflow

```bash
# 1. Train the agent
python train_dqn_discrete.py --timesteps 200000 --seed 42

# 2. Evaluate on standard conditions
python evaluate_dqn_discrete.py \
    --model-path models/dqn_discrete/dqn_discrete_seed42_steps200000.pt \
    --n-episodes 50 \
    --seed 42

# 3. Test generalization with increased rain
python evaluate_dqn_discrete.py \
    --model-path models/dqn_discrete/dqn_discrete_seed42_steps200000.pt \
    --n-episodes 50 \
    --rain-range 0.0 10.0 \
    --seed 123
```

## Comparison with PPO

Both DQN and PPO implementations:
- Use the **same environment dynamics** (water balance, crop stages, climate)
- Use the **same reward function** (state-dependent irrigation cost)
- Report the **same evaluation metrics** for fair comparison
- Share environment parameters by default

Key differences:
- DQN uses **discrete actions** (0-15 mm in integer steps)
- PPO uses **continuous actions** (any value in [0, 15] mm range)
- DQN is off-policy with experience replay
- PPO is on-policy with trajectory rollouts

## Implementation Details

### Observation Processing

The environment returns a Dict observation:
```python
{
    'soil_moisture': array([0.5]),    # 1 dim
    'crop_stage': 1,                  # integer (0, 1, or 2)
    'rain': array([0.3]),             # 1 dim
    'et0': array([0.7])               # 1 dim
}
```

The agent's `obs_to_state()` method:
1. One-hot encodes `crop_stage` → 3 dims
2. Concatenates all features → 6-dim state vector
3. Feeds to Q-network

### Action Discretization

The wrapper maps discrete indices to irrigation amounts:
```python
action_map = [0, 1, 2, 3, ..., 15]  # mm
```

Discrete action `i` → continuous action `i` mm passed to environment.

## Notes

- Uses **Huber loss** for training stability
- **Hard updates** for target network (no soft/polyak averaging)
- No Double DQN, Dueling, or Prioritized Experience Replay
- Simple vanilla DQN for interpretability and simplicity
