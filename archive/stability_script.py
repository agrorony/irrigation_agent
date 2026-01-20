"""
Bin 1 Stability Experiments - Script Version

Run controlled experiments to make soil_bin = 1 dynamically stable.
"""

import numpy as np
from collections import defaultdict
from irrigation_env import IrrigationEnv
from irr_Qtable import discretize_state, N_ACTIONS


def from_discrete_to_components(state_idx, n_soil_bins):
    """Extract state components from discrete state index."""
    rain_bin = state_idx % 2
    et0_bin = (state_idx // 2) % 2
    crop_stage = (state_idx // (2 * 2)) % 3
    soil_bin = state_idx // (3 * 2 * 2)
    return (soil_bin, crop_stage, et0_bin, rain_bin)


def measure_bin1_stability(env, n_episodes=100, n_soil_bins=3, verbose=False):
    """
    Measure residence time in soil_bin = 1 under random policy.
    """
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
        print(f"\\n{'='*60}")
        print(f"BIN 1 STABILITY REPORT ({n_episodes} episodes)")
        print(f"{'='*60}")
        print(f"Total steps: {total_steps}")
        print(f"Bin 1 entries: {entry_count}")
        print(f"Bin 1 time: {bin1_percentage:.1f}%")
        print(f"\\nResidence time in bin 1:")
        print(f"  Mean: {mean_residence:.2f} steps")
        print(f"  Median: {median_residence:.1f} steps")
        print(f"  Max: {max_residence} steps")
        print(f"{'='*60}\\n")
    
    return stats


def run_experiments():
    """Run all stability experiments."""
    
    print("="*70)
    print("SOIL BIN 1 STABILITY EXPERIMENTS")
    print("="*70)
    print("Target: Mean residence time ≥ 10 consecutive steps")
    print("="*70 + "\\n")
    
    results = {}
    
    # Baseline
    print("\\n[1/7] BASELINE")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 40.0),
        max_soil_moisture=100.0, episode_length=90,
    )
    results['Baseline'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # A1: Rain reduction to 20
    print("\\n[2/7] A1: Rain ↓ 20")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 20.0),
        max_soil_moisture=100.0, episode_length=90,
    )
    results['A1 (rain↓20)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # A2: Rain reduction to 10
    print("\\n[3/7] A2: Rain ↓ 10")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 10.0),
        max_soil_moisture=100.0, episode_length=90,
    )
    results['A2 (rain↓10)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # B1: Capacity increase to 150
    print("\\n[4/7] B1: Capacity ↑ 150")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 40.0),
        max_soil_moisture=150.0, episode_length=90,
    )
    results['B1 (cap↑150)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # B2: Capacity increase to 200
    print("\\n[5/7] B2: Capacity ↑ 200")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 40.0),
        max_soil_moisture=200.0, episode_length=90,
    )
    results['B2 (cap↑200)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # D1: Combined moderate
    print("\\n[6/7] D1: Rain ↓ 15 + Cap ↑ 150")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 15.0),
        max_soil_moisture=150.0, episode_length=90,
    )
    results['D1 (rain↓15+cap↑150)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # D2: Combined aggressive
    print("\\n[7/7] D2: Rain ↓ 5 + Cap ↑ 200")
    env = IrrigationEnv(
        max_et0=8.0, max_rain=50.0,
        et0_range=(2.0, 8.0), rain_range=(0.0, 5.0),
        max_soil_moisture=200.0, episode_length=90,
    )
    results['D2 (rain↓5+cap↑200)'] = measure_bin1_stability(env, n_episodes=100, verbose=True)
    
    # Print summary
    print("\\n" + "="*80)
    print("SUMMARY: Mean Residence Time in Bin 1")
    print("="*80)
    print(f"{'Experiment':<25} {'Mean':<10} {'Median':<10} {'Entries':<10} {'Status'}")
    print("-"*80)
    
    for name, stats in results.items():
        mean = stats['mean_residence']
        median = stats['median_residence']
        entries = stats['entry_count']
        status = "✓ STABLE" if mean >= 10 else "✗ Unstable"
        print(f"{name:<25} {mean:<10.2f} {median:<10.1f} {entries:<10d} {status}")
    
    print("="*80)
    
    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['mean_residence'])
    best = results[best_name]
    
    print(f"\\nBEST: {best_name}")
    print(f"  Mean residence: {best['mean_residence']:.2f} steps")
    print(f"  Target reached: {'YES ✓' if best['mean_residence'] >= 10 else 'NO ✗'}")
    
    if best['mean_residence'] < 10:
        gap = 10 - best['mean_residence']
        print(f"  Gap to target: {gap:.2f} steps")
        print(f"\\n  Next steps:")
        print(f"  - Need {gap/best['mean_residence']*100:.1f}% improvement")
        print(f"  - Consider ET strengthening (modify kc values in env)")
        print(f"  - Try rain_range=(0,3) with cap=250")
    
    print("\\n" + "="*80 + "\\n")
    
    return results


if __name__ == "__main__":
    results = run_experiments()
