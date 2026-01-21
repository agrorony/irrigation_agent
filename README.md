# Irrigation Agent: Tabular Q-Learning for Interpretable Irrigation Policy

A reinforcement learning project using **tabular Q-learning** to derive interpretable irrigation scheduling policies under different climatic regimes.
A reinforcement learning project for irrigation scheduling optimization using **tabular Q-learning** as an interpretable policy analysis tool.

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
pip install numpy gymnasium matplotlib scipy stable-baselines3
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

## Continuous Action Space (PPO)

For finer irrigation control beyond the 3 discrete levels (0mm, 5mm, 15mm), use the continuous action environment with PPO. This allows the agent to output **any irrigation amount** in the range [0, 30] mm.

### Quick Start

```python
from irrigation_env_continuous import IrrigationEnvContinuous
from stable_baselines3 import PPO

# Create continuous environment
env = IrrigationEnvContinuous(
    max_et0=8.0,
    max_rain=50.0,
    et0_range=(2.0, 8.0),
    rain_range=(0.0, 0.8),
    max_soil_moisture=320.0,
    episode_length=90,
    max_irrigation=30.0  # Continuous range [0, 30] mm
)

# Train PPO agent
model = PPO("MultiInputPolicy", env, verbose=1, ent_coef=0.01)
model.learn(total_timesteps=200000)

# Evaluate policy
obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print(f"Recommended irrigation: {action[0]:.2f} mm")
```

### Training from Command Line

```bash
# Train continuous PPO (200k timesteps, 4 parallel environments)
python train_ppo_continuous.py --timesteps 200000 --n-envs 4 --seed 42

# Monitor training with TensorBoard
tensorboard --logdir logs/ppo_continuous/

# Evaluate trained model
python evaluate_ppo_continuous.py --n-episodes 50
```

### Key Differences: Discrete vs. Continuous

| Feature | Q-Table (Discrete) | PPO (Continuous) |
|---------|-------------------|------------------|
| **Action space** | 3 actions: 0, 5, 15 mm | Continuous: [0, 30] mm |
| **Interpretability** | ✅ Full transparency (108 Q-values) | ⚠️ Neural network (black box) |
| **Training time** | ~1000 episodes (~10 min) | ~200k timesteps (~2-4 hours) |
| **Granularity** | Coarse (3 irrigation levels) | Fine (any amount, e.g., 12.7 mm) |
| **Sample efficiency** | Excellent (small state space) | Moderate (requires more data) |
| **Use case** | Policy analysis, interpretability | Optimal control, real-world deployment |
| **Exploration** | ε-greedy (simple) | Entropy-regularized (complex) |

### When to Use Continuous Actions

**Use continuous PPO if:**
- ✅ You need fine-grained irrigation control (not just 0/5/15 mm)
- ✅ You're deploying to real irrigation systems with variable flow rates
- ✅ You're okay with neural network "black box" policies
- ✅ You have compute resources for 200k+ timesteps of training

**Use discrete Q-table if:**
- ✅ You need fully interpretable policies (see every decision)
- ✅ You want fast training (<1000 episodes)
- ✅ Coarse irrigation levels (0/5/15 mm) are sufficient
- ✅ You're doing policy analysis or research on RL interpretability

### File Structure

```
irrigation_agent/
├── irrigation_env.py                # Discrete environment (Q-table)
├── irrigation_env_continuous.py     # Continuous environment (PPO)
├── irr_Qtable.py                    # Tabular Q-learning
├── ppo_env.py                       # Environment factory (supports both)
├── train_ppo_continuous.py          # Continuous PPO training
├── evaluate_ppo_continuous.py       # Continuous PPO evaluation
└── models/
    ├── qtable/                      # Q-table policies
    └── ppo_continuous/              # PPO neural network policies
```

### Advanced: Comparing Policies

```python
# Train both approaches on the same regime
from irrigation_env import IrrigationEnv
from irrigation_env_continuous import IrrigationEnvContinuous
from irr_Qtable import train_q_learning
from stable_baselines3 import PPO

# Discrete Q-learning
env_discrete = IrrigationEnv(rain_range=(0.0, 0.8), ...)
Q = train_q_learning(env_discrete, n_episodes=1000)

# Continuous PPO
env_continuous = IrrigationEnvContinuous(rain_range=(0.0, 0.8), max_irrigation=30.0, ...)
model = PPO("MultiInputPolicy", env_continuous)
model.learn(total_timesteps=200000)

# Compare total water usage, rewards, etc.
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
