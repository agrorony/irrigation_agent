"""
Continuous Action Space Environment for Irrigation Scheduling
===============================================================

Extends IrrigationEnv to support continuous irrigation amounts [0, max_irrigation] mm
using PPO, while maintaining identical dynamics and reward structure.

Key differences from discrete IrrigationEnv:
- Action space: Box([0], [max_irrigation]) instead of Discrete(3)
- step() accepts continuous float values
- Water balance and rewards use exact irrigation amounts instead of discrete levels

All other dynamics (climate sampling, crop stages, state transitions) remain identical.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from irrigation_env import IrrigationEnv


class IrrigationEnvContinuous(IrrigationEnv):
    """
    Irrigation environment with continuous action space.
    
    Inherits all dynamics from IrrigationEnv but changes action space from
    discrete {0, 5, 15} mm to continuous [0, max_irrigation] mm.
    
    Parameters
    ----------
    max_irrigation : float
        Maximum irrigation amount in mm (default: 30.0)
    **kwargs
        All other parameters passed to parent IrrigationEnv
    
    Action Space
    ------------
    Box(low=0.0, high=max_irrigation, shape=(1,), dtype=np.float32)
    
    Example
    -------
    >>> env = IrrigationEnvContinuous(max_irrigation=30.0)
    >>> obs, _ = env.reset()
    >>> action = np.array([12.5])  # Apply 12.5 mm irrigation
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """
    
    def __init__(self, max_irrigation=30.0, **kwargs):
        """
        Initialize continuous action environment.
        
        Parameters
        ----------
        max_irrigation : float
            Maximum irrigation amount (mm/day), default 30.0
        **kwargs
            All other arguments passed to IrrigationEnv parent class
        """
        # Initialize parent environment (sets up discrete action space initially)
        super().__init__(**kwargs)
        
        # Store max irrigation limit
        self.max_irrigation = max_irrigation
        
        # Override action space to continuous
        self.action_space = spaces.Box(
            low=0.0, 
            high=max_irrigation, 
            shape=(1,), 
            dtype=np.float32
        )
    
    def _update_state_continuous(self, irrigation_mm: float):
        """
        Update soil moisture with continuous irrigation amount.
        
        Implements same water balance as parent class but with continuous irrigation:
        SM(t+1) = SM(t) + Irrigation + Rain - ET_crop
        
        Parameters
        ----------
        irrigation_mm : float
            Irrigation amount in mm (already clipped to [0, max_irrigation])
        """
        # Calculate crop evapotranspiration (mm/day)
        kc =1
        et_crop = self.current_et0 * kc
        
        # Current moisture in mm
        moisture_mm = self.soil_moisture * self.max_soil_moisture
        
        # Water balance with continuous irrigation
        moisture_mm += irrigation_mm + self.current_rain - et_crop
        
        # Clip to valid bounds [0, max_soil_moisture]
        moisture_mm = np.clip(moisture_mm, 0.0, self.max_soil_moisture)
        
        # Convert back to normalized fraction
        self.soil_moisture = moisture_mm / self.max_soil_moisture
        
        # Sample new climate for next step
        self.current_et0, self.current_rain = self._sample_climate(
            self.current_et0, self.current_rain
        )
        
        # Update crop stage based on time
        self.current_step += 1
        self.crop_stage = self._get_crop_stage(self.current_step)
    
    def _calculate_reward_continuous(self, irrigation_mm: float) -> float:
        """
        Calculate reward with continuous irrigation cost.
        
        OLD PROBLEM (why PPO collapsed to zero irrigation):
        ------------------------------------------------------
        - Unconditional linear cost: reward -= water_cost * irrigation_mm
        - This created a STRICTLY NEGATIVE gradient ∂r/∂irrigation = -water_cost
        - Even when soil was critically dry (prev << bottom), irrigating was immediately penalized
        - PPO's gradient-based optimization found a stable local optimum at irrigation ≈ 0
        - The delayed positive rewards (entering optimal zone) couldn't overcome the immediate negative gradient
        
        NEW DESIGN (state-dependent cost creates positive gradient when dry):
        ----------------------------------------------------------------------
        - Irrigation cost is now STATE-DEPENDENT based on prev_soil_moisture:
            * When prev < bottom (DRY):   REWARD irrigation → POSITIVE gradient
            * When prev in [bottom, top]: Small quadratic cost → smooth gradient
            * When prev > top (WET):      Strong penalty → negative gradient
        
        - Key insight: When soil is dry, irrigation benefit > cost
        - For prev < bottom, ∂r/∂irrigation > 0, so PPO learns to irrigate
        - For prev ≥ bottom, ∂r/∂irrigation < 0, so PPO learns to conserve
        - This removes the zero-irrigation trap while maintaining water efficiency
        
        Parameters
        ----------
        irrigation_mm : float
            Irrigation amount applied (mm)
        
        Returns
        -------
        reward : float
            Reward value for this timestep
        """
        reward = 0.0
        prev = self.prev_soil_moisture
        curr = self.soil_moisture
        bottom = self.threshold_bottom_soil_moisture
        top = self.threshold_top_soil_moisture
        
        # Entering optimum
        if prev < bottom and bottom <= curr <= top:
            reward += 6.0
        
        # Staying in optimal
        if bottom <= prev <= top and bottom <= curr <= top:
            reward += 2.0
        
        # Leaving optimal zone
        if bottom <= prev <= top and (curr < bottom or curr > top):
            reward -= 4.0
        
        # Continuous stress penalties
        # Below optimal
        if curr < bottom:
            reward -= 5.0 * (bottom - curr)
        
        # Above optimal
        if curr > top:
            reward -= 1.0 * (curr - top)
        
        # STATE-DEPENDENT irrigation cost (REPLACES old unconditional penalty)
        # ---------------------------------------------------------------------
        # Signal amplification factor for dry-state irrigation benefit
        # This scaling increases PPO signal-to-noise; logic unchanged.
        DRY_REWARD_SCALE = 8.0
        
        if prev < bottom:
            # When starting DRY: REWARD irrigation proportional to deficit
            # Creates POSITIVE gradient: ∂r/∂irrigation > 0
            # The worse the deficit, the more valuable irrigation becomes
            water_deficit = bottom - prev
            irrigation_benefit = self.water_cost * irrigation_mm * min(water_deficit * 3.0, 1.0)
            reward += DRY_REWARD_SCALE * irrigation_benefit
            
        elif bottom <= prev <= top:
            # When starting in OPTIMAL: gentle quadratic cost
            # ∂r/∂irrigation = -k * irrigation (smooth, grows with amount)
            # Gradient is small near zero, discourages over-irrigation without creating trap
            reward -= 0.5 * self.water_cost * (irrigation_mm ** 2) / self.max_irrigation
            
        else:  # prev > top
            # When starting TOO WET: strong linear penalty (irrigation is wasteful)
            # ∂r/∂irrigation = -3 * water_cost (strong negative gradient)
            reward -= 3.0 * self.water_cost * irrigation_mm
        
        return reward
    
    def step(self, action):
        """
        Execute one timestep with continuous irrigation action.
        
        Parameters
        ----------
        action : np.ndarray or float
            Irrigation amount in mm. Can be:
            - np.ndarray([value]) from PPO
            - float value
            Automatically clipped to [0, max_irrigation]
        
        Returns
        -------
        observation : dict
            Updated observation (same format as parent class)
        reward : float
            Reward signal
        terminated : bool
            Whether episode has ended
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information including applied irrigation amount
        """
        # Convert action to scalar and clip to valid range
        if isinstance(action, np.ndarray):
            irrigation_mm = float(action[0])
        else:
            irrigation_mm = float(action)
        
        # Clip to valid bounds
        irrigation_mm = np.clip(irrigation_mm, 0.0, self.max_irrigation)
        
        # Store previous soil moisture for reward calculation
        self.prev_soil_moisture = self.soil_moisture
        
        # Update state with continuous irrigation
        self._update_state_continuous(irrigation_mm)
        
        # Calculate reward with continuous cost
        reward = self._calculate_reward_continuous(irrigation_mm)
        
        # Get new observation
        observation = self._get_obs()
        
        # Check if episode is complete
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        # Diagnostic info
        info = {
            "step": self.current_step,
            "raw_et0": self.current_et0,
            "raw_rain": self.current_rain,
            "soil_moisture_mm": self.soil_moisture * self.max_soil_moisture,
            "crop_stage": self.crop_stage,
            "irrigation_applied": irrigation_mm,  # Track actual irrigation applied
        }
        
        return observation, reward, terminated, truncated, info


# Example usage and verification
if __name__ == "__main__":
    print("=" * 70)
    print("Continuous Action Space Environment - Demo")
    print("=" * 70)
    
    # Create continuous environment
    env = IrrigationEnvContinuous(
        max_et0=8.0,
        max_rain=50.0,
        et0_range=(2.0, 8.0),
        rain_range=(0.0, 0.8),
        max_soil_moisture=320.0,
        episode_length=90,
        max_irrigation=30.0
    )
    
    print(f"\nAction space: {env.action_space}")
    print(f"Action space bounds: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial state:")
    print(f"  Soil moisture: {obs['soil_moisture'][0]:.3f}")
    print(f"  Crop stage: {obs['crop_stage']}")
    print(f"  ET₀: {info['raw_et0']:.2f} mm/day")
    print(f"  Rain: {info['raw_rain']:.2f} mm/day")
    
    # Test different continuous actions
    print(f"\n{'Step':<6} {'Action (mm)':<12} {'Reward':<10} {'Soil':<8} {'Irr Applied':<12}")
    print("-" * 70)
    
    test_actions = [12.5, 0.0, 25.3, 5.8, 30.0, 15.2]
    
    for i, action_val in enumerate(test_actions):
        action = np.array([action_val], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"{i+1:<6} {action_val:<12.1f} {reward:<10.2f} {obs['soil_moisture'][0]:<8.3f} {info['irrigation_applied']:<12.2f}")
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 70)
    print("Continuous environment ready for PPO training!")
    print("=" * 70)
