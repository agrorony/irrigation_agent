# Irrigation Agent: Tabular Q-Learning for Interpretable Irrigation Policy

A reinforcement learning project using **tabular Q-learning** to derive interpretable irrigation scheduling policies under different climatic regimes.

## Project Overview

This repository implements a discrete-state Q-learning agent for irrigation scheduling, with a focus on **policy interpretability** and **climate-regime analysis** rather than algorithmic advancement. The core contribution is using Q-tables as **interpretable policy artifacts** to understand irrigation strategies under varying rainfall conditions.

### Key Achievements
✅ **Stable Q-learning convergence** on a 36-state discrete environment (12 soil bins × 3 crop stages × 2 ET₀ bins × 2 rain bins)  
✅ **Two climatic regimes trained:** Dry (Rain/ET ≈ 0.09) and Moderate (Rain/ET ≈ 0.33)  
✅ **Full Q-table coverage:** All 36 states learned with deterministic policy extraction  
✅ **Physically meaningful policies:** Interpretable irrigation strategies that adapt to climate  
✅ **Environment calibration:** Stabilized soil moisture dynamics (bin 1 mean residence: 11.28 steps)

### Research Focus
This project uses Q-tables as the **final output**, not as a stepping stone to deep RL. The goal is to:
- Analyze and compare learned irrigation strategies across climatic regimes
- Provide interpretable, actionable insights for irrigation scheduling
- Benchmark against heuristic policies and theoretical baselines

See **[PROJECT_STATUS.md](PROJECT_STATUS.md)** for detailed technical summary, guarantees, limitations, and proposed next steps.

---

## Environment

### State Space (36 discrete states)
- **Soil moisture:** 12 bins over [0, 1] (normalized water content)
- **Crop stage:** 3 stages (0=emergence, 1=flowering, 2=maturity)
- **ET₀ (evapotranspiration):** Binary (low <0.5, high ≥0.5)
- **Rain:** Binary (dry <0.1, wet ≥0.1)

### Action Space (3 discrete actions)
- **Action 0:** No irrigation (0 mm)
- **Action 1:** Light irrigation (5 mm)
- **Action 2:** Heavy irrigation (15 mm)

### Reward Function
- Bonus for maintaining soil moisture in optimal range [0.3, 0.7]
- Penalty for water stress (soil <0.3) and over-irrigation (soil >0.7)
- Cost for irrigation water usage (0.01 per mm)

### Climate Configurations
**Dry Regime (E3+):**
- `rain_range = (0.0, 0.8)` mm/day → Rain/ET ≈ 0.09
- `et0_range = (2.0, 8.0)` mm/day
- `max_soil_moisture = 320.0` mm

**Moderate Rain Regime:**
- `rain_range = (0.0, 3.0)` mm/day → Rain/ET ≈ 0.33
- All other parameters identical

---

## Installation

```bash
pip install numpy gymnasium
```

No additional dependencies required for core Q-learning (matplotlib/seaborn recommended for analysis).

---

## Usage

### Training a Q-Learning Agent
```python
from irrigation_env import IrrigationEnv
from irr_Qtable import train_q_learning, discretize_state

# Create environment (dry regime)
env = IrrigationEnv(
    rain_range=(0.0, 0.8),
    max_soil_moisture=320.0,
    et0_range=(2.0, 8.0),
    episode_length=90
)

# Train Q-learning agent
Q_table = train_q_learning(
    env=env,
    n_episodes=1000,
    alpha=0.1,           # learning rate
    gamma=0.99,          # discount factor
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    n_soil_bins=12
)

# Extract deterministic policy
import numpy as np
policy = np.argmax(Q_table, axis=1)  # policy[state] = best action
```

### Using a Learned Policy
```python
# Deploy policy in environment
observation, info = env.reset()
total_reward = 0

for step in range(90):
    state = discretize_state(observation, n_soil_bins=12)
    action = policy[state]  # deterministic action
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Episode reward: {total_reward}")
```

### Complete Examples
See **[experiments.ipynb](experiments.ipynb)** for:
- Full training runs (dry and moderate regimes)
- Policy extraction and visualization
- Regime comparison analysis
- State coverage experiments

---

## Project Structure

```
irrigation_agent/
├── README.md                   # This file
├── PROJECT_STATUS.md           # Detailed technical status and next steps
├── irrigation_env.py           # Gymnasium-compliant irrigation environment
├── irr_Qtable.py               # Tabular Q-learning implementation
├── dp_solver.py                # Dynamic programming baseline (value iteration)
├── experiments.ipynb           # Training runs and initial analysis
├── irrigation_env.ipynb        # Environment testing and validation
└── archive/                    # Historical calibration experiments
    ├── STABILITY_REPORT.md     # Soil bin 1 stability experiments
    └── (calibration scripts)
```

---

## Key Results

### Dry Regime Policy (Rain/ET = 0.09)
- **Irrigation frequency:** 19.4% of states recommend irrigation
- **Strategy:** Aggressive irrigation when dry (soil bin 0), conservative when moist (bins 1-2)
- **Convergence:** Stable after ~500 episodes (mean reward ~177)

### Moderate Rain Regime Policy (Rain/ET = 0.33)
- **Irrigation frequency:** 11.1% of states recommend irrigation (−8.3 pp vs. dry)
- **Strategy:** Relies more on natural rainfall, especially in rain-present states (27.8% → 5.6% irrigation)
- **Adaptation:** Policy correctly reduces water application in response to increased rainfall

### Physical Interpretation
Both policies are **agronomically sound:**
- No irrigation when soil is saturated (bin 2)
- Stage-dependent irrigation (higher during flowering)
- Rain-reactive behavior (less irrigation when rain present)

---

## Next Steps (Proposed)

See **[PROJECT_STATUS.md](PROJECT_STATUS.md)** Section 3 for detailed proposals:

1. **Comparative Regime Analysis:** Quantify policy divergence, visualize decision boundaries, simulate cross-regime deployment
2. **Robustness Testing:** Out-of-distribution evaluation, perturbation analysis, hyperparameter sensitivity
3. **Benchmark Against Baselines:** Compare to threshold heuristics, fixed schedules, and DP-optimal policy

---

## Constraints and Design Philosophy

- ✅ **Tabular Q-learning only** (no neural networks, DQN, PPO, etc.)
- ✅ **Interpretability first:** Q-tables as final research output, not intermediate step
- ✅ **Fixed environment dynamics:** No drainage, runoff, or reward modifications
- ✅ **Physical calibration:** Stability achieved through parameter tuning, not algorithm tweaks

---

## Citation

If you use this work, please cite:
```
@misc{irrigation_qtable_2026,
  author = {agrorony},
  title = {Tabular Q-Learning for Interpretable Irrigation Scheduling},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/agrorony/irrigation_agent}
}
```

---

## License

TBD

---

## Contact

For questions or collaboration: See repository issues or discussions.
