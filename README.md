# Irrigation Agent

A reinforcement learning project for irrigation scheduling optimization using **tabular Q-learning** as an interpretable policy analysis tool.

## Overview

This project implements **discrete state-action Q-learning** for irrigation scheduling with:
- **State space**: 36 discrete states (12 soil bins × 3 crop stages × 2 ET₀ bins × 2 rain bins)
- **Action space**: 3 irrigation levels (0mm, 5mm, 15mm)
- **Approach**: Tabular Q-learning (no neural networks, no function approximation)
- **Purpose**: Generate **interpretable Q-tables** for policy analysis and regime comparison

## Project Status

✅ **Physical stability calibration** completed (soil bin 1: 11.28 step residence)  
✅ **Q-learning convergence** achieved under dry regime (Rain/ET ≈ 0.09)  
✅ **Regime shift experiment** completed (moderate rainfall: Rain/ET ≈ 0.33)  
✅ **Policy extraction** successful (deterministic policies via argmax)  
✅ **Interpretability** validated (policies align with agronomic principles)

📄 **See [PROJECT_STATUS.md](PROJECT_STATUS.md) for comprehensive technical summary, Q-table capabilities, and proposed next steps.**

## Key Achievements

### 1. Stable Q-Learning Under Arid Conditions
- Converged after ~500 episodes (mean reward: ~177)
- Conservative policy: 80.6% no irrigation, 8.3% light, 11.1% heavy
- Physically interpretable: rain-aware, moisture-responsive, crop-stage-sensitive

### 2. Climate-Adaptive Policy Learning
- Moderate rainfall regime (Rain/ET = 0.33) learned to reduce irrigation by 43%
- Both regimes converged to similar rewards despite different strategies
- Demonstrates climate-adaptive behavior without manual rule engineering

### 3. Full Transparency
- 36×3 Q-table (108 values) is human-inspectable
- Policies extractable as deterministic lookup tables
- No "black box" neural networks—every decision is traceable

## Files

- **`irr_Qtable.py`** - Core Q-learning implementation (state discretization, training loop, policy extraction)
- **`irrigation_env.py`** - Gymnasium-compliant environment with configurable climate parameters
- **`experiments.ipynb`** - Full experimental workflow (stability calibration, training, regime comparison)
- **`PROJECT_STATUS.md`** - **[START HERE]** Comprehensive research status and next steps
- **`archive/`** - Historical stability experiments and validation scripts

## Installation

```bash
pip install numpy gymnasium matplotlib scipy
```

## Quick Start

```python
from irrigation_env import IrrigationEnv
from irr_Qtable import train_q_learning, discretize_state
import numpy as np

# Create environment (dry regime configuration)
env = IrrigationEnv(
    max_et0=8.0,
    max_rain=50.0,
    et0_range=(2.0, 8.0),
    rain_range=(0.0, 0.8),
    max_soil_moisture=320.0,
    episode_length=90
)

# Train Q-learning agent
Q = train_q_learning(env, n_episodes=1000, alpha=0.1, gamma=0.99)

# Extract policy
policy = np.argmax(Q, axis=1)  # Best action for each state
print(f"Learned policy: {policy}")
```

## Research Use

This project is designed for **academic analysis of tabular RL policies**, not as a production irrigation controller. Use cases include:

- **Comparative regime analysis** - How do policies differ across climates?
- **Interpretability studies** - Can we explain why certain actions are chosen?
- **Baseline benchmarking** - How do Q-learning policies compare to heuristics?
- **Robustness testing** - Are policies stable across hyperparameters?

See [PROJECT_STATUS.md](PROJECT_STATUS.md) Section 3 for detailed next-step proposals.

## Key Constraints

🚫 **No neural networks** - Tabular methods only (interpretability > scalability)  
🚫 **No environment modifications** - Fixed dynamics, fixed rewards  
🚫 **No reward engineering** - Simple yield - water_cost formula maintained  
✅ **Focus on analysis** - Extraction, comparison, visualization, documentation

## Citation

If you use this work, please cite:

```
Irrigation Agent: Tabular Q-Learning for Climate-Adaptive Irrigation Scheduling
Agrorony Research, 2026
https://github.com/agrorony/irrigation_agent
```

## License

TBD
