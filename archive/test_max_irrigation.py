"""
Minimal verification test for max_irrigation parameter propagation.

Tests that max_irrigation is correctly wired through to the action space.
"""

from ppo_env import make_irrigation_env

print("=" * 80)
print("VERIFICATION TEST: max_irrigation parameter propagation")
print("=" * 80)

# Test Case 1: max_irrigation = 15.0
print("\nTest Case 1: max_irrigation = 15.0")
print("-" * 80)
env1 = make_irrigation_env(
    seed=0,
    continuous=True,
    max_irrigation=15.0,
    max_et0=15.0,
    max_rain=5.0,
    max_soil_moisture=100.0
)
print(f"RESULT: action_space = {env1.action_space}")
print(f"Expected: Box(0.0, 15.0, (1,), float32)")
assert env1.action_space.low[0] == 0.0, "Lower bound should be 0.0"
assert env1.action_space.high[0] == 15.0, "Upper bound should be 15.0"
print("✓ PASS: Action space matches max_irrigation=15.0")

# Test Case 2: max_irrigation = 20.0
print("\nTest Case 2: max_irrigation = 20.0")
print("-" * 80)
env2 = make_irrigation_env(
    seed=0,
    continuous=True,
    max_irrigation=20.0,
    max_et0=15.0,
    max_rain=5.0,
    max_soil_moisture=100.0
)
print(f"RESULT: action_space = {env2.action_space}")
print(f"Expected: Box(0.0, 20.0, (1,), float32)")
assert env2.action_space.low[0] == 0.0, "Lower bound should be 0.0"
assert env2.action_space.high[0] == 20.0, "Upper bound should be 20.0"
print("✓ PASS: Action space matches max_irrigation=20.0")

# Test Case 3: Default max_irrigation = 30.0
print("\nTest Case 3: Default max_irrigation (should be 30.0)")
print("-" * 80)
env3 = make_irrigation_env(
    seed=0,
    continuous=True,
    max_et0=15.0,
    max_rain=5.0,
    max_soil_moisture=100.0
)
print(f"RESULT: action_space = {env3.action_space}")
print(f"Expected: Box(0.0, 30.0, (1,), float32)")
assert env3.action_space.low[0] == 0.0, "Lower bound should be 0.0"
assert env3.action_space.high[0] == 30.0, "Upper bound should be 30.0"
print("✓ PASS: Action space matches default max_irrigation=30.0")

print("\n" + "=" * 80)
print("ALL TESTS PASSED ✓")
print("max_irrigation parameter is correctly wired to action space!")
print("=" * 80)
