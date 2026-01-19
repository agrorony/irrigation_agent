"""
E3+ Configuration: Ultra-Stable Version

Slightly more aggressive than E3 to ensure 100% success rate.
"""

import numpy as np
from collections import defaultdict
from irrigation_env import IrrigationEnv
from irr_Qtable import discretize_state, N_ACTIONS


def from_discrete_to_components(state_idx, n_soil_bins):
    rain_bin = state_idx % 2
    et0_bin = (state_idx // 2) % 2
    crop_stage = (state_idx // (2 * 2)) % 3
    soil_bin = state_idx // (3 * 2 * 2)
    return (soil_bin, crop_stage, et0_bin, rain_bin)


def measure_bin1_stability(env, n_episodes=100, n_soil_bins=3):
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
    else:
        mean_residence = 0
        median_residence = 0
        max_residence = 0
    
    total_steps = sum(bin_visits.values())
    bin1_percentage = 100 * bin_visits[1] / total_steps if total_steps > 0 else 0
    
    return {
        'mean_residence': mean_residence,
        'median_residence': median_residence,
        'max_residence': max_residence,
        'bin1_percentage': bin1_percentage,
    }


def test_configurations():
    """Test E3 vs E3+ configurations."""
    
    print("="*70)
    print("CONFIGURATION COMPARISON: E3 vs E3+")
    print("="*70)
    print("Testing both configurations across 5 trials each")
    print("="*70 + "\\n")
    
    configs = {
        'E3': {
            'rain_range': (0.0, 1.0),
            'max_soil_moisture': 300.0,
            'description': 'Original successful config'
        },
        'E3+': {
            'rain_range': (0.0, 0.8),
            'max_soil_moisture': 320.0,
            'description': 'Enhanced for 100% reliability'
        }
    }
    
    all_results = {}
    
    for config_name, config_params in configs.items():
        print(f"\\nTesting {config_name}: {config_params['description']}")
        print(f"  rain_range={config_params['rain_range']}")
        print(f"  max_soil_moisture={config_params['max_soil_moisture']}")
        print()
        
        results = []
        for trial in range(1, 6):
            print(f"  Trial {trial}/5...", end=" ", flush=True)
            
            env = IrrigationEnv(
                max_et0=8.0,
                max_rain=50.0,
                et0_range=(2.0, 8.0),
                rain_range=config_params['rain_range'],
                max_soil_moisture=config_params['max_soil_moisture'],
                episode_length=90,
            )
            
            stats = measure_bin1_stability(env, n_episodes=100)
            results.append(stats)
            
            status = "✓" if stats['mean_residence'] >= 10 else "✗"
            print(f"Mean={stats['mean_residence']:.2f} {status}")
        
        all_results[config_name] = results
    
    # Compare
    print("\\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"{'Config':<10} {'Mean±SD':<15} {'Min':<8} {'Max':<8} {'Success%':<12} {'Reliability'}")
    print("-"*70)
    
    for config_name, results in all_results.items():
        means = [r['mean_residence'] for r in results]
        mean_of_means = np.mean(means)
        std_of_means = np.std(means)
        min_mean = min(means)
        max_mean = max(means)
        successes = sum(1 for m in means if m >= 10)
        success_rate = 100 * successes / len(results)
        
        if success_rate == 100:
            reliability = "★★★ Perfect"
        elif success_rate >= 80:
            reliability = "★★ Good"
        else:
            reliability = "★ Needs work"
        
        print(f"{config_name:<10} {mean_of_means:5.2f}±{std_of_means:4.2f}    " +
              f"{min_mean:6.2f}  {max_mean:6.2f}  {success_rate:6.0f}%      {reliability}")
    
    print("="*70)
    
    # Recommendation
    best_config = None
    best_rate = 0
    
    for config_name, results in all_results.items():
        means = [r['mean_residence'] for r in results]
        success_rate = 100 * sum(1 for m in means if m >= 10) / len(results)
        
        if success_rate > best_rate:
            best_rate = success_rate
            best_config = config_name
    
    print(f"\\n★ RECOMMENDED CONFIGURATION: {best_config}")
    
    if best_config:
        params = configs[best_config]
        means = [r['mean_residence'] for r in all_results[best_config]]
        
        print(f"\\nFinal Configuration for Production:")
        print(f"```python")
        print(f"env = IrrigationEnv(")
        print(f"    max_et0=8.0,")
        print(f"    max_rain=50.0,")
        print(f"    et0_range=(2.0, 8.0),")
        print(f"    rain_range={params['rain_range']},")
        print(f"    max_soil_moisture={params['max_soil_moisture']},")
        print(f"    episode_length=90,")
        print(f")")
        print(f"```")
        print(f"\\nPerformance:")
        print(f"  - Average mean residence: {np.mean(means):.2f} steps")
        print(f"  - Success rate: {best_rate:.0f}%")
        print(f"  - Bin 1 is DYNAMICALLY STABLE")
    
    print("\\n" + "="*70 + "\\n")


if __name__ == "__main__":
    test_configurations()
