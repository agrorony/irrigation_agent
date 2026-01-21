"""
Final Validation: E3 Configuration Stability

Verify that the successful E3 configuration consistently achieves
mean residence ≥ 10 steps across multiple independent runs.
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


def measure_bin1_stability(env, n_episodes=100, n_soil_bins=3):
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
    
    if len(residence_times) > 0:
        mean_residence = np.mean(residence_times)
        median_residence = np.median(residence_times)
        max_residence = np.max(residence_times)
        std_residence = np.std(residence_times)
    else:
        mean_residence = 0
        median_residence = 0
        max_residence = 0
        std_residence = 0
    
    total_steps = sum(bin_visits.values())
    bin1_percentage = 100 * bin_visits[1] / total_steps if total_steps > 0 else 0
    
    return {
        'residence_times': residence_times,
        'mean_residence': mean_residence,
        'median_residence': median_residence,
        'max_residence': max_residence,
        'std_residence': std_residence,
        'entry_count': entry_count,
        'bin_visits': dict(bin_visits),
        'bin1_percentage': bin1_percentage,
    }


def validate_e3_configuration():
    """Run multiple independent validations of E3 configuration."""
    
    print("="*70)
    print("FINAL VALIDATION: E3 Configuration")
    print("="*70)
    print("Configuration: rain_range=(0, 1) + max_soil_moisture=300")
    print("Target: Mean residence ≥ 10 steps")
    print("Validation: 5 independent runs of 100 episodes each")
    print("="*70 + "\\n")
    
    # Run 5 independent trials
    n_trials = 5
    results = []
    
    for trial in range(1, n_trials + 1):
        print(f"Trial {trial}/{n_trials}...", end=" ", flush=True)
        
        env = IrrigationEnv(
            max_et0=8.0,
            max_rain=50.0,
            et0_range=(2.0, 8.0),
            rain_range=(0.0, 1.0),        # E3 parameter
            max_soil_moisture=300.0,      # E3 parameter
            episode_length=90,
        )
        
        stats = measure_bin1_stability(env, n_episodes=100, n_soil_bins=3)
        results.append(stats)
        
        status = "✓" if stats['mean_residence'] >= 10 else "✗"
        print(f"Mean={stats['mean_residence']:.2f} {status}")
    
    # Aggregate results
    mean_residences = [r['mean_residence'] for r in results]
    median_residences = [r['median_residence'] for r in results]
    max_residences = [r['max_residence'] for r in results]
    bin1_percentages = [r['bin1_percentage'] for r in results]
    
    print("\\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(f"{'Trial':<10} {'Mean':<10} {'Median':<10} {'Max':<8} {'Bin1%':<10} {'Status'}")
    print("-"*70)
    
    for i, stats in enumerate(results, 1):
        status = "✓ PASS" if stats['mean_residence'] >= 10 else "✗ FAIL"
        print(f"Trial {i:<4} {stats['mean_residence']:<10.2f} " +
              f"{stats['median_residence']:<10.1f} {stats['max_residence']:<8d} " +
              f"{stats['bin1_percentage']:<10.1f} {status}")
    
    print("-"*70)
    print(f"{'AVERAGE':<10} {np.mean(mean_residences):<10.2f} " +
          f"{np.mean(median_residences):<10.1f} {np.mean(max_residences):<8.1f} " +
          f"{np.mean(bin1_percentages):<10.1f}")
    print(f"{'STD DEV':<10} {np.std(mean_residences):<10.2f} " +
          f"{np.std(median_residences):<10.1f} {np.std(max_residences):<8.1f} " +
          f"{np.std(bin1_percentages):<10.1f}")
    
    print("="*70)
    
    # Success rate
    successes = sum(1 for m in mean_residences if m >= 10)
    success_rate = 100 * successes / n_trials
    
    print(f"\\nSUCCESS RATE: {successes}/{n_trials} ({success_rate:.0f}%)")
    print(f"Average mean residence: {np.mean(mean_residences):.2f} ± {np.std(mean_residences):.2f}")
    print(f"Average bin 1 occupancy: {np.mean(bin1_percentages):.1f}%")
    
    if success_rate == 100:
        print("\\n✓✓✓ VALIDATION SUCCESSFUL ✓✓✓")
        print("E3 configuration CONSISTENTLY achieves target stability")
        print("Bin 1 is DYNAMICALLY STABLE")
    elif success_rate >= 80:
        print("\\n✓ VALIDATION MOSTLY SUCCESSFUL")
        print(f"Configuration achieves target in {success_rate:.0f}% of runs")
        print("Variability may require slightly more aggressive parameters")
    else:
        print("\\n✗ VALIDATION FAILED")
        print("Configuration does not consistently achieve target")
        print("Requires parameter adjustment or ET weakening")
    
    print("\\n" + "="*70)
    
    # Detailed statistics
    print("\\nDETAILED STATISTICS:")
    print(f"  Mean residence time:")
    print(f"    Range: [{min(mean_residences):.2f}, {max(mean_residences):.2f}]")
    print(f"    95% CI: [{np.mean(mean_residences) - 1.96*np.std(mean_residences):.2f}, " +
          f"{np.mean(mean_residences) + 1.96*np.std(mean_residences):.2f}]")
    print(f"  Median residence time: {np.mean(median_residences):.1f} ± {np.std(median_residences):.1f}")
    print(f"  Maximum observed residence: {max(max_residences)} steps")
    print(f"  Bin 1 occupancy: {np.mean(bin1_percentages):.1f}% ± {np.std(bin1_percentages):.1f}%")
    
    print("\\n" + "="*70 + "\\n")
    
    return results


if __name__ == "__main__":
    results = validate_e3_configuration()
