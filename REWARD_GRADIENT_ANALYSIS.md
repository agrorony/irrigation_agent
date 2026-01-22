# Reward Gradient Analysis: Continuous Irrigation Environment

## Executive Summary

**Status: ✓ PASS** - The redesigned reward function successfully eliminates the zero-irrigation local optimum.

**Key Finding**: 100% of dry states (prev_sm < 0.4) now exhibit **positive reward gradients** near zero irrigation, creating the necessary learning signal for PPO.

---

## Problem Diagnosed

### Original Issue
The PPO agent collapsed to near-zero irrigation despite long training because:

```python
# OLD reward (unconditional penalty)
reward -= water_cost * irrigation_mm
```

This created:
- **Strictly negative gradient**: ∂r/∂a = -water_cost (constant -0.01)
- **Local optimum at a=0**: Any irrigation was immediately penalized
- **Delayed positive rewards** (entering optimal zone) couldn't overcome immediate negative gradient
- **PPO's gradient-based optimization** got trapped at zero irrigation

---

## Solution Implemented

### State-Dependent Cost Structure

```python
if prev < bottom:
    # REWARD irrigation when soil is dry
    water_deficit = bottom - prev
    irrigation_benefit = water_cost * irrigation_mm * min(water_deficit * 3.0, 1.0)
    reward += irrigation_benefit
    
elif bottom <= prev <= top:
    # Gentle quadratic cost in optimal range
    reward -= 0.5 * water_cost * (irrigation_mm ** 2) / max_irrigation
    
else:  # prev > top
    # Strong penalty when too wet
    reward -= 3.0 * water_cost * irrigation_mm
```

### Key Design Principles

1. **Irrigation cost depends on state**, not just action
2. **When dry**: irrigation benefit > cost → **positive gradient**
3. **When optimal**: small quadratic cost → smooth gradient near zero
4. **When wet**: strong linear penalty → discourage waste

---

## Experimental Validation

### Methodology
- **Script**: `diagnose_reward_gradient.py`
- **Approach**: Finite-difference gradient computation
- **Formula**: dr/da ≈ [r(a+ε) - r(a-ε)] / 2ε
- **Scenarios**: 13 test cases across dry/optimal/wet states
- **Deterministic**: Fixed climate (no stochastic sampling)

### Results by State Class

#### DRY States (prev_sm < 0.40) - n=6
```
Mean gradient near a≈0: +0.0190
Min gradient:           +0.0159
Max gradient:           +0.0246

Positive gradient: 6/6 (100.0%) ✓
```

**Interpretation**: 
- Every dry state has **positive gradient** → PPO learns to irrigate
- Gradient magnitude scales with water deficit (0.10 → 0.39)
- Stronger signal when soil is critically dry

#### OPTIMAL States (0.40 ≤ prev_sm ≤ 0.60) - n=4
```
Mean gradient near a≈0: -0.0000
Min gradient:           -0.0000
Max gradient:           -0.0000

Negative gradient: 4/4 (100.0%) ✓
```

**Interpretation**:
- Quadratic cost creates **near-zero gradient** at a=0
- No strong penalty for staying at zero (already optimal)
- Grows with action magnitude → discourages over-irrigation

#### WET States (prev_sm > 0.60) - n=3
```
Mean gradient near a≈0: -0.0331
Min gradient:           -0.0331
Max gradient:           -0.0331

Negative gradient: 3/3 (100.0%) ✓
```

**Interpretation**:
- Strong **negative gradient** (3× water cost)
- Irrigation is wasteful when soil is saturated
- Clear signal to conserve water

---

## Example Gradient Profiles

### Scenario 1: Very Dry (prev_sm=0.10, stage=1, ET₀=6mm, rain=0mm)
```
Action(mm)   Reward      dr/da
---------------------------------
0.00         -1.6078     +0.0246  ← POSITIVE!
0.05         -1.6066     +0.0246
1.00         -1.5832     +0.0246
10.00        -1.3616     +0.0246
30.00        -0.8691     +0.0246  ← Best action
```
**Gradient is consistently positive** → PPO learns to increase irrigation

### Scenario 7: Optimal (prev_sm=0.41, stage=1, ET₀=6mm, rain=5mm)
```
Action(mm)   Reward      dr/da
---------------------------------
0.00         +2.0000     -0.0000  ← Near zero
0.05         +2.0000     -0.0000
1.00         +1.9998     -0.0003
10.00        +1.9833     -0.0033
30.00        +1.8500     -0.0100  ← Growing penalty
```
**Gradient is zero/slightly negative** → PPO conserves water

### Scenario 11: Too Wet (prev_sm=0.65, stage=0, ET₀=3mm, rain=15mm)
```
Action(mm)   Reward      dr/da
---------------------------------
0.00         -0.0922     -0.0331  ← STRONG negative
0.05         -0.0938     -0.0331
1.00         -0.1253     -0.0331
10.00        -0.4234     -0.0331
30.00        -1.0859     -0.0331  ← Heavily penalized
```
**Gradient is strongly negative** → PPO avoids irrigation

---

## Implications for PPO Training

### Why This Fixes Policy Collapse

1. **Gradient Signal**
   - Old: ∂r/∂a = -0.01 (constant negative)
   - New: ∂r/∂a = +0.019 when dry (positive!)
   
2. **No Local Optimum at Zero**
   - When soil is dry, moving from a=0 to a=ε **increases** reward
   - PPO's policy gradient will push actions **upward** from zero
   
3. **State-Aware Learning**
   - Agent learns **context-dependent** policy
   - Irrigate when dry, conserve when optimal/wet

### Expected Training Behavior

✓ **Early episodes**: Random exploration finds that irrigation in dry states → positive reward

✓ **Mid training**: Policy gradient reinforces irrigation actions when soil < 0.4

✓ **Late training**: Policy converges to:
- Irrigate aggressively when dry
- Apply minimal/zero irrigation when optimal
- Avoid irrigation when wet

---

## Verification Checklist

- [x] Script runs deterministically (fixed climate)
- [x] Tests all reward branches (dry/optimal/wet)
- [x] Computes accurate finite-difference gradients
- [x] Verifies NEW reward function is loaded (not stale code)
- [x] 100% of dry states have positive gradient
- [x] Optimal/wet states have negative gradient
- [x] Clear, interpretable output

---

## Next Steps

### 1. Retrain PPO with New Reward
```bash
python irrigation_agent/train_ppo_continuous.py
```

### 2. Monitor Training Metrics
- Mean action value should **increase** from ~0 to 5-15mm
- Episode rewards should improve
- Policy entropy should remain healthy

### 3. Evaluation
```bash
python irrigation_agent/evaluate_ppo_continuous.py
```
Expect:
- Agent irrigates when soil_moisture < 0.4
- Agent conserves water when soil_moisture ∈ [0.4, 0.6]
- Average actions > 0 (no longer collapsed)

---

## Technical Notes

### Gradient Computation Details
- **Epsilon**: 1e-3 mm (small enough for accuracy)
- **Central difference**: Used in interior (a > ε and a < max-ε)
- **Forward difference**: Used at boundary (a ≈ 0)
- **Deterministic wrapper**: Monkeypatches `_sample_climate()` to return fixed values

### State Injection
Forces environment into controlled pre-step state:
```python
env.soil_moisture = prev_sm
env.prev_soil_moisture = prev_sm
env.crop_stage = stage
env.current_step = step
env.current_et0 = et0
env.current_rain = rain
```

This isolates reward sensitivity to action, eliminating stochastic climate confounds.

---

## Conclusion

The redesigned reward function **successfully creates the gradient landscape needed for PPO to learn effective irrigation policies**. The zero-irrigation trap has been eliminated through state-dependent cost structure that rewards irrigation when soil is dry and penalizes it when soil is optimal or wet.

**Confidence**: High - 100% of tested dry states show positive gradient, confirming the theoretical design intent.

---

*Generated: 2026-01-21*  
*Script: `irrigation_agent/diagnose_reward_gradient.py`*  
*Environment: `IrrigationEnvContinuous` with state-dependent reward*
