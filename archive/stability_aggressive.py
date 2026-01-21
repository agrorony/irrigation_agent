"""
Bin 1 Stability Experiments - AGGRESSIVE PHASE

Testing extreme parameter combinations to reach target of 10 steps mean residence.
"""

import numpy as np
from collections import defaultdict
from irrigation_agent.irrigation_env import IrrigationEnv
from irr_Qtable import discretize_state, N_ACTIONS


def from_discrete_to_components(state_idx, n_soil_bins):
    """Extract state components from discrete state index."""
    rain_bin = state_idx % 2
    et0_bin = (state_idx // 2) % 2
    crop_stage = (state_idx // (2 * 2)) % 3
    soil_bin = state_idx // (3 * 2 * 2)
    return (soil_bin, crop_stage, et0_bin, rain_bin)


def measure_bin1_stability(env, n_episodes=100, n_soil_bins=3, verbose=False):
    """Measure residence time in soil_bin = 1 under random policy."""
    residence_times = []
    bin_visits = defaultdict(int)
    entry_count = 0
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        in_bin1 = False
        current_residence = 0
        
        while not done:
            state_idx = discretize_state(obs, n_soil_bins)
            soil_bin, _, _, _ = from_discrete_to_components(state_idx, n_soil_bins)
            
            bin_visits[soil_bin] += 1
            
            if soil_bin == 1:
                if not in_bin1:
                    in_bin1 = True
                    entry_count += 1
                    current_residence = 1
                else:
                    current_residence += 1
            else:
                if in_bin1:
                    residence_times.append(current_residence)
                    in_bin1 = False
                    current_residence = 0
            
            action = np.random.randint(N_ACTIONS)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        if in_bin1:
            residence_times.append(current_residence)
    
    # Calculate statistics
    if len(residence_times) > 0:
        mean_residence = np.mean(residence_times)
        median_residence = np.median(residence_times)
        max_residence = np.max(residence_times)
        min_residence = np.min(residence_times)
    else:
        mean_residence = 0
        median_residence = 0
        max_residence = 0
        min_residence = 0
    
    total_steps = sum(bin_visits.values())
    bin1_percentage = 100 * bin_visits[1] / total_steps if total_steps > 0 else 0
    
    stats = {
        'residence_times': residence_times,
        'mean_residence': mean_residence,
        'median_residence': median_residence,
        'max_residence': max_residence,
        'min_residence': min_residence,
        'entry_count': entry_count,
        'bin_visits': dict(bin_visits),
        'bin1_percentage': bin1_percentage,
        'total_steps': total_steps,
        'n_episodes': n_episodes,
    }
    
    if verbose:
        print(f"\\nBin 1: entries={entry_count}, mean={mean_residence:.2f}, " +
              f"median={median_residence:.1f}, max={max_residence}, " +
              f"time%={bin1_percentage:.1f}%")
    
    return stats


def run_aggressive_experiments():
    """Run aggressive parameter experiments."""
    
    print("="*70)
    print("AGGRESSIVE STABILITY EXPERIMENTS")
    print("="*70)
    print("Target: Mean residence ≥ 10 steps in bin 1")
    print("="*70 + "\\n")
    
    results = {}
    
    # E1: Very low rain + very high capacity
    print("[E1] Rain ↓ 3 + Cap ↑ 250", end="")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 3.0),
        max_soil_moisture=250.0, episode_length=90,
    )
    results['E1 (rain↓3+cap↑250)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # E2: Minimal rain + very high capacity
    print("[E2] Rain ↓ 2 + Cap ↑ 250", end="")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 2.0),
        max_soil_moisture=250.0, episode_length=90,
    )
    results['E2 (rain↓2+cap↑250)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # E3: Near-zero rain + extreme capacity
    print("[E3] Rain ↓ 1 + Cap ↑ 300", end="")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 1.0),
        max_soil_moisture=300.0, episode_length=90,
    )
    results['E3 (rain↓1+cap↑300)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # E4: Reduce ET range + low rain
    print("[E4] ET ↓ (1-5) + Rain ↓ 3 + Cap ↑ 200", end="")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(1.0, 5.0),  # Reduced ET
        rain_range=(0.0, 3.0),
        max_soil_moisture=200.0, episode_length=90,
    )
    results['E4 (ET↓+rain↓3+cap↑200)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # E5: Very reduced ET + moderate rain reduction
    print("[E5] ET ↓ (1-4) + Rain ↓ 5 + Cap ↑ 200", end="")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(1.0, 4.0),  # Very low ET
        rain_range=(0.0, 5.0),
        max_soil_moisture=200.0, episode_length=90,
    )
    results['E5 (ET↓↓+rain↓5+cap↑200)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # Print summary
    print("\\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Experiment':<30} {'Mean':<8} {'Median':<8} {'Max':<6} {'Status'}")
    print("-"*70)
    
    for name, stats in results.items():
        mean = stats['mean_residence']
        median = stats['median_residence']
        max_res = stats['max_residence']
        status = "✓ SUCCESS" if mean >= 10 else f"Gap: {10-mean:.1f}"
        print(f"{name:<30} {mean:<8.2f} {median:<8.1f} {max_res:<6d} {status}")
    
    print("="*70)
    
    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['mean_residence'])
    best = results[best_name]
    
    print(f"\\n★ BEST: {best_name}")
    print(f"  Mean residence: {best['mean_residence']:.2f} steps")
    print(f"  Median residence: {best['median_residence']:.1f} steps")
    print(f"  Max residence: {best['max_residence']} steps")
    print(f"  Bin 1 time: {best['bin1_percentage']:.1f}%")
    
    if best['mean_residence'] >= 10:
        print(f"\\n  ✓✓✓ TARGET ACHIEVED! ✓✓✓")
        print(f"  Bin 1 is now DYNAMICALLY STABLE")
    else:
        gap = 10 - best['mean_residence']
        print(f"\\n  Gap to target: {gap:.2f} steps ({gap/10*100:.1f}% remaining)")
        
        if best['mean_residence'] >= 8:
            print(f"  → VERY CLOSE! Consider:")
            print(f"     - rain_range=(0, 0.5) with current capacity")
            print(f"     - or increase capacity to 350mm")
        elif best['mean_residence'] >= 6:
            print(f"  → Getting there. Consider:")
            print(f"     - More aggressive rain reduction")
            print(f"     - ET weakening (reduce kc values)")
        else:
            print(f"  → Need ET weakening in environment code")
            print(f"     - Modify kc_by_stage values in irrigation_env.py")
            print(f"     - Current kc: [0.5, 1.15, 0.7]")
            print(f"     - Try: [0.4, 0.9, 0.5] or lower")
    
    print("\\n" + "="*70 + "\\n")
    
    return results


if __name__ == "__main__":
    results = run_aggressive_experiments()
