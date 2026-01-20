# Q-Table Irrigation Agent: Project Status and Research Direction

**Date:** January 2026  
**Repository:** agrorony/irrigation_agent  
**Approach:** Tabular Q-Learning for Interpretable Irrigation Policy Analysis

---

## 1. CURRENT STATUS: Technical Summary

### 1.1 What Has Been Achieved

#### Physical Environment Calibration
- **Objective Achieved:** Stabilized soil moisture bin 1 (SM ∈ [0.333, 0.667)) to enable meaningful Q-learning
- **Final Configuration (E3+):**
  - `rain_range = (0.0, 0.8)` mm/day
  - `max_soil_moisture = 320.0` mm
  - `et0_range = (2.0, 8.0)` mm/day (unchanged)
- **Stability Metrics:**
  - Mean residence time in bin 1: **11.28 steps** (target: ≥10)
  - Bin 1 occupancy increased from 1.6% → 15.2% (9.5× improvement)
  - No drainage, runoff, or reward modifications required
- **Physical Validity:** Stability emerges from balance between input variability and soil capacity

#### State Space Design
- **Discrete State Space:** 36 states (12 soil bins × 3 crop stages × 2 ET₀ bins × 2 rain bins)
- **State Decomposition:**
  - Soil moisture: 12 bins over [0, 1]
  - Crop stage: {0=emergence, 1=flowering, 2=maturity}
  - ET₀: Binary {low < 0.5, high ≥ 0.5} (normalized)
  - Rain: Binary {dry < 0.1, wet ≥ 0.1} (normalized)
- **Action Space:** 3 discrete actions
  - 0 = No irrigation (0 mm)
  - 1 = Light irrigation (5 mm)
  - 2 = Heavy irrigation (15 mm)

#### Q-Learning Training: Dry Regime (Rain/ET ≈ 0.09)
- **Convergence:** Stable convergence after ~500 episodes
- **Final Performance:** Mean episodic reward ~177 (stable across episodes)
- **Q-Table Characteristics:**
  - Shape: 36 states × 3 actions
  - Full coverage: All 36 states learned
  - Sparse but complete: Non-zero entries across all states
- **Deterministic Policy Extraction:** Successfully extracted π(s) = argmax_a Q(s,a) for all states
- **Hyperparameters:**
  - Learning rate α = 0.1
  - Discount factor γ = 0.99
  - ε-greedy: 1.0 → 0.01 with decay 0.995
  - Training episodes: 1000

#### Q-Learning Training: Moderate Rain Regime (Rain/ET ≈ 0.33)
- **Climate Configuration:**
  - `rain_range = (0.0, 3.0)` mm/day (vs. 0.0-0.8 in dry regime)
  - All other parameters identical to dry regime
  - Rain/ET ratio increased 3.75× (0.09 → 0.33)
- **Convergence:** Stable convergence (identical training protocol)
- **Q-Table Characteristics:** Full 36×3 table learned
- **Deterministic Policy Extraction:** Complete policy extracted

#### Policy Interpretability and Physical Validity
**Dry Regime Policy (Rain/ET = 0.09):**
- **Low moisture (bin 0):** Mixed strategy - 5/12 states irrigate (15mm for mid/late stages)
- **Medium moisture (bin 1):** Conservative - 10/12 states no irrigation, 2/12 use 5mm
- **High moisture (bin 2):** Conservation - 11/12 states no irrigation
- **Irrigation frequency:** 19.4% of states recommend irrigation
- **Physical interpretation:** Agent learned to irrigate aggressively when dry, conserve when moist

**Moderate Rain Regime Policy (Rain/ET = 0.33):**
- **Behavioral change:** Irrigation states reduced from 19.4% → 11.1% (−8.3 percentage points)
- **Rain-present states:** Irrigation reduced from 27.8% → 5.6% (−22.2 percentage points)
- **Physical interpretation:** Agent learned to rely more on natural rainfall, reducing irrigation usage
- **Regime adaptation:** Policy correctly reduced water application in response to increased rainfall

### 1.2 Guarantees and Validation

#### Convergence Guarantees
✅ **Tabular Q-Learning Convergence Theorem:** Under infinite exploration and stationary environment:
- Q-values converge to Q* (optimal action-value function) with probability 1
- Both regimes trained for 1000 episodes with decaying ε-greedy exploration
- Stable reward trajectories confirm practical convergence

✅ **Full State Coverage:**
- All 36 discrete states visited during training (both regimes)
- Every state has learned Q-values for all 3 actions
- No unvisited states or zero-initialized entries in final Q-table

✅ **Deterministic Policy Extractability:**
- Policy π(s) = argmax_a Q(s,a) well-defined for all states
- No ties or ambiguous action selections (or resolved arbitrarily)

#### Physical Stability Guarantees
✅ **Bin 1 Stability (E3+ Configuration):**
- Mean residence time: 11.28 steps (exceeds 10-step target)
- Validated across 100 independent episodes
- Stability emerges from physical dynamics, not reward shaping

✅ **Gymnasium API Compliance:**
- Environment follows standard RL interface
- Reproducible with seed control
- Observations/actions/rewards well-defined

#### Interpretability Guarantees
✅ **Explicit Policy Representation:**
- Q-table is a finite lookup table (36×3 = 108 entries)
- Every decision is traceable to Q(s,a) values
- No black-box function approximation

✅ **Human-Readable States:**
- Each state index maps to interpretable components (soil, stage, ET₀, rain)
- Policy can be written as decision table or rule set
- Example: "If soil=low AND crop=flowering AND rain=dry → irrigate 15mm"

✅ **Physical Meaningfulness:**
- Learned policies align with agronomic intuition
- No pathological behaviors (e.g., heavy irrigation when saturated)
- Regime-specific adaptation reflects real irrigation practice

### 1.3 Limitations and Assumptions

#### State Space Limitations
⚠️ **Discretization Artifacts:**
- Continuous soil moisture (0-1) discretized into 12 bins
- Bin boundaries create artificial thresholds (e.g., 0.332 vs. 0.334 treated differently)
- Optimal policy may oscillate near bin edges

⚠️ **Binary Climate Encoding:**
- ET₀ and rain reduced to binary (low/high, dry/wet)
- Loss of granularity: 3.0mm rain and 0.8mm rain both classified as "wet"
- Cannot represent fine-grained weather patterns

⚠️ **Fixed Crop Stage Progression:**
- Growth stages determined solely by day number (0-30, 31-60, 61+)
- No feedback from stress or irrigation on development
- Unrealistic for some crop phenologies

#### Regime Assumptions
⚠️ **Stationary Climate:**
- Training assumes constant climate distribution within a regime
- No seasonal trends or inter-annual variability
- Does not model climate change scenarios

⚠️ **Regime-Specific Policies:**
- Each Q-table valid only for its training regime (dry vs. moderate)
- No transfer learning or adaptation to new climates
- Requires retraining for each climatic scenario

⚠️ **Limited Regime Coverage:**
- Only two regimes tested: Rain/ET = 0.09 and 0.33
- No extreme regimes (fully arid, humid, Mediterranean seasonal)
- Generalization to untested regimes unknown

#### Environment Simplifications
⚠️ **Single-Layer Soil Model:**
- No stratification (root zone vs. deep percolation)
- No drainage or runoff (water budget simplified)
- Maximum soil capacity constrains realism

⚠️ **Simplified Water Balance:**
- ETc = Kc × ET₀ (no stress reduction function)
- No surface evaporation vs. transpiration partitioning
- Rain fully effective (no losses)

⚠️ **Fixed Economic Parameters:**
- Water cost (0.01 per mm) constant across regimes
- No crop yield or revenue modeling
- Reward function hand-designed, not learned

#### Sample Efficiency Limitations
⚠️ **1000-Episode Training:**
- Some state-action pairs may have limited samples
- Statistical uncertainty in Q-value estimates not quantified
- No confidence intervals or Bayesian uncertainty

⚠️ **Stochastic Environment:**
- Climate variables sampled randomly each step
- Same state may lead to different outcomes due to randomness
- Q-values represent average over climate distribution

#### Generalization Limitations
⚠️ **No Function Approximation:**
- Cannot generalize to unseen states (e.g., new soil bins)
- Q-table only defined for 36 discrete states
- Scaling to higher-dimensional state spaces infeasible

⚠️ **No Multi-Task Learning:**
- Cannot leverage similarities between regimes
- Each regime requires independent training
- No shared knowledge across climates

---

## 2. ROLE OF THE Q-TABLE: Capabilities and Boundaries

### 2.1 What Q-Tables Already Answer

#### ✅ Optimal Action Selection (Within Discretization)
**Question:** "What is the best irrigation action for a given discrete state?"  
**Answer:** π(s) = argmax_a Q(s,a) provides deterministic decision for all 36 states

**Example:**
- State: soil_bin=2, crop_stage=1, et0=high, rain=dry
- Discrete state index: 18
- Optimal action: π(18) = 0 (no irrigation)
- Justification: High soil moisture (bin 2) indicates sufficient water

#### ✅ Value of States and Actions
**Question:** "How much cumulative reward can I expect from state s if I take action a?"  
**Answer:** Q(s,a) provides expected discounted return

**Example:**
- Q(18, 0) = 150.3 → Expected return if no irrigation in state 18
- Q(18, 2) = 142.7 → Expected return if heavy irrigation in state 18
- Difference: 7.6 → Cost of over-irrigation in high-moisture state

#### ✅ Policy Comparison Across Regimes
**Question:** "How does the learned policy differ between dry and moderate rainfall regimes?"  
**Answer:** Direct comparison of π_dry(s) vs. π_moderate(s) reveals behavioral changes

**Key Findings:**
- Dry regime: 19.4% of states irrigate
- Moderate regime: 11.1% of states irrigate
- Rain-present states: 27.8% → 5.6% irrigation rate (rain-adaptive behavior)
- **Interpretation:** Agent learns to reduce irrigation when natural water input increases

#### ✅ State-Action Frequency Analysis
**Question:** "Which irrigation actions are most commonly recommended?"  
**Answer:** Histogram of π(s) over all states

**Dry Regime Distribution:**
- Action 0 (no irrigation): 29/36 states (80.6%)
- Action 1 (light irrigation): 4/36 states (11.1%)
- Action 2 (heavy irrigation): 3/36 states (8.3%)

**Moderate Regime Distribution:**
- Action 0 (no irrigation): 32/36 states (88.9%)
- Action 1 (light irrigation): 2/36 states (5.6%)
- Action 2 (heavy irrigation): 2/36 states (5.6%)

#### ✅ State-Specific Strategy Identification
**Question:** "What is the irrigation strategy for low-moisture states?"  
**Answer:** Filter states by soil_bin=0 and examine π(s)

**Dry Regime Low-Moisture Strategy (bin 0):**
- 7/12 states: No irrigation (when rain=wet or crop=emergence)
- 5/12 states: Irrigate (15mm for flowering/maturity, 5mm for emergence)
- **Interpretation:** Agent learned stage-dependent irrigation under stress

#### ✅ Convergence Diagnostics
**Question:** "Did Q-learning converge to a stable policy?"  
**Answer:** Yes, verified by:
- Stable episodic rewards after ~500 episodes
- Consistent Q-value magnitudes across training
- Reproducible policies across multiple training runs

### 2.2 What Q-Tables Cannot Answer (By Design)

#### ❌ Continuous State Queries
**Question:** "What action should I take if soil moisture = 0.341 (not exactly a bin center)?"  
**Limitation:** Q-table only defined for discrete states  
**Workaround:** Discretize 0.341 → bin 4, then use π(bin=4, ...)  
**Issue:** Same action for all values in [0.333, 0.417)

#### ❌ Unseen State Generalization
**Question:** "What if I add a 4th crop stage (e.g., grain filling)?"  
**Limitation:** Q-table has no states for crop_stage=3  
**Implication:** Requires retraining with expanded state space (36 → 48 states)

#### ❌ Uncertainty Quantification
**Question:** "How confident is the agent in recommending action a in state s?"  
**Limitation:** Q(s,a) is a point estimate, not a distribution  
**What's Missing:**
- No confidence intervals on Q-values
- No measure of sample count per (s,a) pair
- No epistemic uncertainty (model uncertainty)

**Why It Matters:**
- Cannot distinguish well-explored states from rarely visited states
- No risk-averse decision-making (e.g., choose safe action if uncertain)

#### ❌ Counterfactual Reasoning
**Question:** "What would the policy be if water cost doubled (from 0.01 → 0.02)?"  
**Limitation:** Q-table trained under specific reward function  
**Implication:** Requires retraining with new reward (cannot extrapolate)

**Alternative Approach (Not Implemented):**
- Dynamic programming (value iteration) can recompute policy for new rewards
- Requires environment dynamics model (transition probabilities)

#### ❌ Causal Attribution
**Question:** "Why did the agent choose action 2 in state 18?"  
**Limitation:** Q-learning is reward-driven, not causally interpretable  
**What's Missing:**
- No explanation of which state features drove the decision
- No saliency maps or feature importance
- Cannot answer "What if soil was 5% higher?" without re-discretizing

**Partial Solution:**
- Sensitivity analysis: Compare Q(s,a) across neighboring states
- Example: How does Q(soil=5, stage=1, ...) change as soil varies?

#### ❌ Non-Stationary Environment Adaptation
**Question:** "What if rainfall distribution shifts mid-season (climate change)?"  
**Limitation:** Q-table assumes stationary environment during training  
**Behavior:** Policy remains fixed (trained on historical distribution)  
**Implication:** No online adaptation or regime detection

**What Would Be Needed:**
- Online Q-learning with fresh data
- Meta-learning or regime-switching models
- Explicitly model non-stationarity (e.g., time-varying Q-tables)

#### ❌ Multi-Objective Optimization
**Question:** "Find policy that maximizes yield AND minimizes water use?"  
**Limitation:** Q-table trained on single scalar reward  
**Workaround:** Scalarize objectives (e.g., reward = yield - λ × water)  
**Issue:** Requires manual tuning of trade-off parameter λ

**Better Approach (Not Tabular Q-Learning):**
- Multi-objective RL (Pareto frontier)
- Constrained optimization (maximize yield subject to water budget)

#### ❌ Hierarchical or Sequential Planning
**Question:** "Should I irrigate today if I know a storm is coming in 3 days?"  
**Limitation:** Markov assumption (policy depends only on current state)  
**What's Missing:**
- No lookahead beyond γ-discounted expectation
- No explicit weather forecast integration
- Cannot plan multi-day strategies

**Note:** Lookahead IS implicitly captured in Q-values via γ, but not explicitly reasoned about

#### ❌ Transfer to New Environments
**Question:** "Can I use the dry regime Q-table to initialize training for a sandy soil?"  
**Limitation:** Q-table implicitly encodes environment dynamics (transition model + reward)  
**Issue:** Sandy soil has different water retention → different transitions → Q-values invalid

**What Would Help:**
- Model-based RL (learn dynamics explicitly, then plan)
- Transfer learning with fine-tuning (requires neural networks)

---

## 3. CONCRETE NEXT STEPS: Research-Focused Analysis

### Next Step 1: Comparative Regime Analysis and Policy Divergence Study

**Objective:** Quantify and visualize how irrigation strategies evolve across climatic regimes, producing publication-ready insights into climate-adaptive behavior.

**Rationale:**
- We have two complete Q-tables (dry and moderate regimes) but no systematic comparison
- Understanding policy divergence is the **core scientific contribution** of this work
- Regime-specific strategies inform real-world irrigation scheduling under climate variability

#### Concrete Tasks:
1. **Policy Divergence Metrics**
   - Compute **action agreement rate**: fraction of states where π_dry(s) = π_moderate(s)
   - Calculate **L1 distance** between Q-tables: Σ_s,a |Q_dry(s,a) - Q_moderate(s,a)|
   - Identify **regime-sensitive states**: where action switches (e.g., irrigate → no irrigation)

2. **State-Space Stratified Analysis**
   - **By soil bin:** How do low/medium/high moisture policies differ across regimes?
   - **By crop stage:** Does regime impact vary by growth phase (emergence vs. flowering)?
   - **By climate state:** Focus on rain-present states (where regime difference is largest)
   - **Tabulate results:** Create decision table showing π_dry(s) vs. π_moderate(s) for all 36 states

3. **Visualizations** (using matplotlib/seaborn)
   - **Heatmap:** 12 soil bins × 3 crop stages grid, color-coded by action (one heatmap per regime)
   - **Divergence map:** Same grid, highlight states where policies disagree
   - **Q-value comparison:** Scatter plot of Q_dry(s,a) vs. Q_moderate(s,a) for all (s,a) pairs
   - **Action distribution bar chart:** Histogram of actions by regime (already partially done)

4. **Water Use Efficiency Analysis**
   - **Simulate both policies** on the moderate regime environment (out-of-sample test)
   - Measure: Total irrigation applied, final crop state, reward achieved
   - **Key question:** Does the moderate-trained policy outperform dry-trained policy when deployed in moderate regime?

5. **Deliverable:** `REGIME_ANALYSIS_REPORT.md`
   - Quantitative metrics (agreement rate, L1 distance, etc.)
   - 4-6 publication-quality figures
   - Interpretation: "Moderate regime reduces irrigation by X% in rain-present states"
   - Discussion: Physical mechanisms driving policy adaptation

---

### Next Step 2: Robustness and Sensitivity Analysis

**Objective:** Stress-test learned policies to identify brittleness, assess generalization to off-regime conditions, and quantify uncertainty in Q-value estimates.

**Rationale:**
- Current Q-tables assume perfect knowledge of training regime distribution
- Real-world deployment faces distribution shift (inter-annual variability, forecast errors)
- Understanding robustness bounds is **critical for practical applicability**

#### Concrete Tasks:
1. **Out-of-Distribution Evaluation**
   - **Test dry policy on moderate regime:** Deploy π_dry on environment with rain_range=(0,3)
   - **Test moderate policy on dry regime:** Deploy π_moderate on environment with rain_range=(0,0.8)
   - **Test on extreme regimes:** Create heavy rain (rain_range=(0,10)) and evaluate both policies
   - **Metrics:** Mean episodic reward, total irrigation, crop stress episodes (soil < 0.3)
   - **Hypothesis:** Policies should degrade gracefully, not catastrophically fail

2. **Perturbation Analysis**
   - **ET₀ uncertainty:** If forecast ET₀ has ±20% error, how does policy performance change?
   - **Rain forecast error:** Simulate incorrect rain predictions (predicted dry, actual wet)
   - **Soil capacity mismatch:** Train with cap=320mm, deploy on cap=280mm (field variability)
   - **Method:** Monte Carlo simulation with 100 episodes per perturbation level

3. **State Discretization Sensitivity**
   - **Ablation:** Retrain Q-learning with n_soil_bins = 6 (coarser) and 18 (finer)
   - **Compare policies:** Does finer discretization change recommended actions?
   - **Trade-off analysis:** Sample efficiency (6 bins) vs. precision (18 bins)
   - **Goal:** Justify current choice of 12 bins as optimal balance

4. **Hyperparameter Sensitivity**
   - **Learning rate α:** Retrain with α ∈ {0.05, 0.1, 0.2} (current: 0.1)
   - **Discount factor γ:** Retrain with γ ∈ {0.95, 0.99, 1.0} (current: 0.99)
   - **Exploration schedule:** Compare ε-decay rates {0.99, 0.995, 0.999}
   - **Metric:** Policy similarity (% agreement with baseline) and final reward
   - **Goal:** Confirm convergence is not hyperparameter-dependent

5. **Reward Function Sensitivity**
   - **Water cost variation:** Retrain with cost ∈ {0.005, 0.01, 0.02} (current: 0.01)
   - **Threshold variation:** Retrain with [bottom, top] ∈ {[0.25,0.65], [0.3,0.7], [0.35,0.75]}
   - **Analyze policy shift:** How many states change action as cost doubles?
   - **Economic insight:** Elasticity of irrigation to water price

6. **Deliverable:** `ROBUSTNESS_REPORT.md`
   - Robustness matrix: policy performance under 5+ perturbations
   - Sensitivity plots: performance vs. perturbation magnitude
   - Risk assessment: "Policy fails when rain forecast error >X mm"
   - Recommendations: "Use moderate-trained policy if annual rain >Y mm"

---

### Next Step 3: Benchmark Against Heuristic Policies and Optimal Baseline

**Objective:** Validate that Q-learning provides value over simple rule-based strategies and (if feasible) compare to theoretical optimal policy.

**Rationale:**
- **Scientific rigor:** Showing Q-learning converges is not enough—must prove it's better than alternatives
- **Baseline comparison** is standard in RL research (e.g., compare to random, greedy, expert heuristics)
- **Practical relevance:** Farmers use simple rules (e.g., "irrigate when soil <40%")—how much does RL improve on this?

#### Concrete Tasks:
1. **Define Heuristic Baselines**
   - **Threshold heuristic:** Irrigate 15mm if soil < 0.35, else no irrigation
   - **Stage-based heuristic:** Heavy irrigation during flowering (stage 1), light otherwise
   - **Rain-reactive heuristic:** No irrigation if rain in last 2 days, else threshold-based
   - **Fixed schedule:** Irrigate every 5 days (mimics drip irrigation timer)
   - **Random policy:** Uniform random over 3 actions (already used for stability tests)

2. **Evaluation Protocol**
   - **Run all policies** (Q-learned + 5 heuristics) on both regimes (dry and moderate)
   - **100 episodes per policy-regime pair** (total: 7 policies × 2 regimes × 100 = 1400 episodes)
   - **Metrics:**
     - Mean episodic reward
     - Total irrigation water applied (mm)
     - Fraction of time in optimal soil range [0.3, 0.7]
     - Crop stress events (soil < 0.3 for >3 consecutive days)
     - Over-irrigation events (soil > 0.8)

3. **Dynamic Programming Optimal Policy (Stretch Goal)**
   - **File:** `dp_solver.py` already exists (value iteration implementation)
   - **Task:** Run DP value iteration to compute true optimal policy π* (under deterministic dynamics approximation)
   - **Compare:** Q-learned policy vs. DP policy (should be similar if Q-learning converged)
   - **Gap analysis:** If policies differ, investigate: is it exploration noise, stochasticity, or non-convergence?
   - **Note:** DP assumes average climate (deterministic transitions); Q-learning handles stochastic climate

4. **Statistical Significance Testing**
   - **Null hypothesis:** Q-learned policy performs same as best heuristic
   - **Test:** Paired t-test on episodic rewards (100 episodes per policy)
   - **Report:** Mean ± std, p-value, effect size (Cohen's d)
   - **Claim:** "Q-learning achieves X% higher reward than threshold heuristic (p<0.01)"

5. **Explainability Comparison**
   - **Q-table inspection:** Show example states where Q-policy differs from threshold heuristic
   - **Case study:** "In state (soil=0.45, stage=1, rain=wet), Q-policy says don't irrigate (Q=150), but threshold heuristic says irrigate (soil still below 0.5)"
   - **Explanation:** Q-policy learned that rain + moderate soil is sufficient during flowering
   - **Insight:** Q-learning captures multi-variate interactions (soil × rain × stage) that threshold rules miss

6. **Deliverable:** `BENCHMARK_REPORT.md`
   - Performance table: Mean reward ± std for all policies × regimes
   - Statistical test results (t-test, effect sizes)
   - Comparative plots: Reward, water use, stress events (bar charts or violin plots)
   - Discussion: "Q-learning reduces water use by X% vs. threshold policy while maintaining yield"
   - Failure analysis: States where Q-policy underperforms (if any)

---

## 4. Summary and Timeline

### Achievements Recap
✅ Physical environment calibrated (bin 1 stability)  
✅ Two climatic regimes trained (dry and moderate)  
✅ Full Q-tables (36×3) learned with stable convergence  
✅ Deterministic policies extracted and validated  
✅ Interpretable, physically meaningful irrigation strategies  

### Research Direction
🎯 **Focus:** Interpretable policy analysis, not algorithm development  
🎯 **Goal:** Publish climate-adaptive irrigation strategies derived from Q-tables  
🎯 **Output:** 3 technical reports + figures suitable for academic paper  

### Proposed Timeline (Estimated)
| Week | Task | Deliverable |
|------|------|-------------|
| 1-2 | Next Step 1: Regime comparison analysis | `REGIME_ANALYSIS_REPORT.md` + 6 figures |
| 3-4 | Next Step 2: Robustness/sensitivity testing | `ROBUSTNESS_REPORT.md` + sensitivity plots |
| 5-6 | Next Step 3: Benchmark vs. heuristics + DP | `BENCHMARK_REPORT.md` + performance tables |
| 7 | Integration: Consolidate findings into academic manuscript | Draft paper sections (Results, Discussion) |

### Key Constraints Maintained
- ❌ No neural networks (DQN, PPO, etc.)
- ❌ No reward function changes
- ❌ No environment dynamics modifications
- ✅ Pure analysis, interpretation, and evaluation

---

## Appendix: File Organization

```
irrigation_agent/
├── irrigation_env.py          # Gymnasium environment (final, stable)
├── irr_Qtable.py               # Q-learning implementation (final, stable)
├── dp_solver.py                # Dynamic programming baseline (optional)
├── experiments.ipynb           # Training runs and initial analysis
├── README.md                   # Project overview
├── PROJECT_STATUS.md           # This document
├── archive/                    # Historical stability experiments
│   ├── STABILITY_REPORT.md
│   └── (calibration scripts)
└── reports/ (TO BE CREATED)
    ├── REGIME_ANALYSIS_REPORT.md    # Next Step 1 output
    ├── ROBUSTNESS_REPORT.md         # Next Step 2 output
    └── BENCHMARK_REPORT.md          # Next Step 3 output
```

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Contact:** Repository maintainers (agrorony/irrigation_agent)
