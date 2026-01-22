"""
Reward Function Sanity Check - Deterministic Offline Test

Verifies that for DRY soil (below threshold), higher irrigation produces
strictly higher immediate reward.

This test MUST FAIL if the reward function is misaligned.
"""

import numpy as np
from irrigation_env_continuous import IrrigationEnvContinuous


def run_sanity_check():
    """
    Test reward monotonicity for dry soil state.
    """
    print("=" * 80)
    print("REWARD FUNCTION SANITY CHECK")
    print("=" * 80)
    print("\nObjective: Verify that higher irrigation → higher reward when soil is DRY")
    print("Method: Fixed state, deterministic environment, three irrigation levels")
    
    # Create deterministic environment
    print("\nSetting up deterministic environment...")
    env = IrrigationEnvContinuous(
        episode_length=1,
        rain_range=(0.0, 0.0),        # No rain
        et0_range=(5.0, 5.0),          # Fixed ET0
        max_soil_moisture=100.0,
        threshold_bottom_soil_moisture=0.4,
        threshold_top_soil_moisture=0.7,
        max_irrigation=15.0,
        max_et0=15.0,
        max_rain=5.0
    )
    
    # Fixed test state (DRY soil)
    TEST_SOIL_MOISTURE = 0.30  # Below threshold_bottom (0.4)
    TEST_CROP_STAGE = 1
    TEST_RAIN = 0.0
    TEST_ET0 = 5.0
    
    print(f"\nFixed test state:")
    print(f"  Soil moisture: {TEST_SOIL_MOISTURE} (threshold_bottom={env.threshold_bottom_soil_moisture})")
    print(f"  → State: DRY (below threshold)")
    print(f"  Crop stage: {TEST_CROP_STAGE}")
    print(f"  Rain: {TEST_RAIN} mm/day")
    print(f"  ET₀: {TEST_ET0} mm/day")
    
    # Test actions
    test_actions = [0.0, 5.0, 10.0]
    
    print(f"\nTesting actions: {test_actions} mm")
    print("-" * 80)
    
    # Collect rewards for each action
    rewards = []
    
    for action_mm in test_actions:
        # Reset to fixed state
        env.reset(seed=42)
        
        # Manually override to fixed state
        env.soil_moisture = TEST_SOIL_MOISTURE
        env.prev_soil_moisture = TEST_SOIL_MOISTURE
        env.crop_stage = TEST_CROP_STAGE
        env.current_rain = TEST_RAIN
        env.current_et0 = TEST_ET0
        env.current_step = 0
        
        # Take action and record reward
        action = np.array([action_mm], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        
        print(f"Action: {action_mm:5.1f} mm  →  Reward: {reward:8.4f}")
    
    # Print results table
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\n{'Action (mm)':<15} {'Reward':<15}")
    print("-" * 30)
    for action, reward in zip(test_actions, rewards):
        print(f"{action:<15.1f} {reward:<15.4f}")
    
    # Verify monotonicity
    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    
    reward_0 = rewards[0]
    reward_5 = rewards[1]
    reward_10 = rewards[2]
    
    print(f"\nExpected behavior (DRY soil):")
    print(f"  reward(10.0) > reward(5.0) > reward(0.0)")
    print(f"\nActual:")
    print(f"  reward(10.0) = {reward_10:.4f}")
    print(f"  reward(5.0)  = {reward_5:.4f}")
    print(f"  reward(0.0)  = {reward_0:.4f}")
    
    # Check conditions
    checks = []
    
    print(f"\nCondition checks:")
    
    # Check 1: reward(10) > reward(5)
    check1 = reward_10 > reward_5
    checks.append(check1)
    symbol1 = "✓" if check1 else "✗"
    print(f"  {symbol1} reward(10.0) > reward(5.0): {reward_10:.4f} > {reward_5:.4f} = {check1}")
    
    # Check 2: reward(5) > reward(0)
    check2 = reward_5 > reward_0
    checks.append(check2)
    symbol2 = "✓" if check2 else "✗"
    print(f"  {symbol2} reward(5.0) > reward(0.0): {reward_5:.4f} > {reward_0:.4f} = {check2}")
    
    # Overall result
    all_passed = all(checks)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ SANITY CHECK PASSED")
        print("  The reward function correctly incentivizes irrigation for DRY soil.")
    else:
        print("✗ SANITY CHECK FAILED")
        print("  The reward function does NOT properly incentivize irrigation!")
        print("\n  PROBLEM: When soil is DRY (below threshold), higher irrigation")
        print("           should yield higher reward, but it doesn't.")
        print("\n  This indicates the reward function is misaligned.")
        raise AssertionError(
            f"Reward monotonicity violated for DRY soil!\n"
            f"  Expected: reward(10.0) > reward(5.0) > reward(0.0)\n"
            f"  Got: {reward_10:.4f}, {reward_5:.4f}, {reward_0:.4f}"
        )
    
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    try:
        run_sanity_check()
    except AssertionError as e:
        print(f"\n{e}")
        exit(1)
