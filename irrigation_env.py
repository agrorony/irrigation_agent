"""
Gymnasium Environment for Irrigation Scheduling
================================================

A reinforcement learning environment for irrigation decision-making with:
- Configurable climatic regimes (ET₀ and rainfall ranges)
- Single-layer soil moisture dynamics
- Discrete crop growth stages
- Full Gymnasium API compliance

Design: Climate parameters are constructor arguments, not hard-coded constants.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IrrigationEnv(gym.Env):
    """
    Irrigation scheduling environment with configurable climate parameters.
    
    State Space:
        - soil_moisture: [0, 1], normalized water content in root zone
        - crop_stage: {0, 1, 2}, discrete growth stage (emergence/flowering/maturity)
        - rain: [0, 1], normalized effective rainfall (mm/day)
        - et0: [0, 1], normalized reference evapotranspiration (mm/day)
    
    Action Space:
        - 0: No irrigation
        - 1: Light irrigation
        - 2: Heavy irrigation
    
    Parameters
    ----------
    max_et0 : float
        Maximum ET₀ value for normalization (mm/day)
    max_rain : float
        Maximum rainfall value for normalization (mm/day)
    et0_range : tuple of float, optional
        (min, max) range for sampling ET₀ values (mm/day)
    rain_range : tuple of float, optional
        (min, max) range for sampling rainfall values (mm/day)
    max_soil_moisture : float, optional
        Maximum soil water holding capacity (mm), default 100
    initial_soil_moisture : float, optional
        Starting soil moisture as fraction [0, 1], default 0.5
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        max_et0: float = 8.0,
        max_rain: float = 50.0,
        et0_range: tuple = (2.0, 8.0),
        rain_range: tuple = (0.0, 40.0),
        max_soil_moisture: float = 100.0,
        initial_soil_moisture: float = 0.5,
        episode_length: int = 90,
    ):
        super().__init__()
        
        # Store climate configuration parameters
        self.max_et0 = max_et0  # For normalization (mm/day)
        self.max_rain = max_rain  # For normalization (mm/day)
        self.et0_range = et0_range  # Sampling range (mm/day)
        self.rain_range = rain_range  # Sampling range (mm/day)
        
        # Soil parameters
        self.max_soil_moisture = max_soil_moisture  # Maximum water capacity (mm)
        self.initial_soil_moisture = initial_soil_moisture  # Initial moisture fraction [0, 1]
        
        # Episode parameters
        self.episode_length = episode_length  # Days per episode
        self.current_step = 0
        
        # Irrigation amounts (mm/day)
        self.irrigation_amounts = np.array([0.0, 5.0, 15.0])
        
        # Crop coefficient by growth stage (for ET calculation)
        # Kc values: emergence (low), flowering (high), maturity (medium)
        self.kc_by_stage = np.array([0.5, 1.15, 0.7])
        
        # Reward function parameters
        self.water_cost = 0.1  # Cost per mm of irrigation applied
        
        # Define observation space using Dict for clarity
        self.observation_space = spaces.Dict({
            "soil_moisture": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "crop_stage": spaces.Discrete(3),  # 0=emergence, 1=flowering, 2=maturity
            "rain": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "et0": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        })
        
        # Define action space
        self.action_space = spaces.Discrete(3)  # 0=no irrigation, 1=light, 2=heavy
        
        # Current state variables
        self.soil_moisture = None  # Fraction [0, 1]
        self.crop_stage = None  # Integer {0, 1, 2}
        self.current_rain = None  # mm/day (raw value)
        self.current_et0 = None  # mm/day (raw value)
    
    def _sample_climate(self) -> tuple:
        """
        Sample climatic conditions from configured ranges.
        
        Returns
        -------
        et0 : float
            Reference evapotranspiration (mm/day)
        rain : float
            Effective rainfall (mm/day)
        """
        # Sample from uniform distribution within configured ranges
        et0 = np.random.uniform(self.et0_range[0], self.et0_range[1])
        rain = np.random.uniform(self.rain_range[0], self.rain_range[1])
        
        return et0, rain
    
    def _normalize_climate(self, et0: float, rain: float) -> tuple:
        """
        Normalize climate variables to [0, 1] using configured maxima.
        
        Parameters
        ----------
        et0 : float
            Raw ET₀ value (mm/day)
        rain : float
            Raw rainfall value (mm/day)
        
        Returns
        -------
        et0_norm : float
            Normalized ET₀ [0, 1]
        rain_norm : float
            Normalized rainfall [0, 1]
        """
        et0_norm = np.clip(et0 / self.max_et0, 0.0, 1.0)
        rain_norm = np.clip(rain / self.max_rain, 0.0, 1.0)
        
        return et0_norm, rain_norm
    
    def _get_crop_stage(self, day: int) -> int:
        """
        Determine crop growth stage based on day of season.
        
        Parameters
        ----------
        day : int
            Current day in episode [0, episode_length)
        
        Returns
        -------
        stage : int
            Crop stage: 0=emergence, 1=flowering, 2=maturity
        """
        # Simple stage progression: divide season into thirds
        stage_duration = self.episode_length // 3
        
        if day < stage_duration:
            return 0  # Emergence
        elif day < 2 * stage_duration:
            return 1  # Flowering
        else:
            return 2  # Maturity
    
    def _update_state(self, action: int):
        """
        Update soil moisture and other state variables based on action and climate.
        
        Water balance equation:
        SM(t+1) = SM(t) + Irrigation(action) + Rain - ET_crop
        
        Where:
        - ET_crop = ET₀ × Kc(crop_stage)
        - All values constrained to valid bounds
        
        Parameters
        ----------
        action : int
            Irrigation decision {0, 1, 2}
        """
        # Get irrigation amount for this action (mm)
        irrigation = self.irrigation_amounts[action]
        
        # Calculate crop evapotranspiration (mm/day)
        kc = self.kc_by_stage[self.crop_stage]
        et_crop = self.current_et0 * kc
        
        # Update soil moisture (mm)
        # Current moisture in mm
        moisture_mm = self.soil_moisture * self.max_soil_moisture
        
        # Water balance
        moisture_mm += irrigation + self.current_rain - et_crop
        
        # Clip to valid bounds [0, max_soil_moisture]
        moisture_mm = np.clip(moisture_mm, 0.0, self.max_soil_moisture)
        
        # Convert back to normalized fraction
        self.soil_moisture = moisture_mm / self.max_soil_moisture
        
        # Sample new climate for next step
        self.current_et0, self.current_rain = self._sample_climate()
        
        # Update crop stage based on time
        self.current_step += 1
        self.crop_stage = self._get_crop_stage(self.current_step)
    
    def _get_obs(self) -> dict:
        """
        Get current observation in Gymnasium format.
        
        Returns
        -------
        obs : dict
            Dictionary containing normalized state variables
        """
        # Normalize climate variables
        et0_norm, rain_norm = self._normalize_climate(self.current_et0, self.current_rain)
        
        return {
            "soil_moisture": np.array([self.soil_moisture], dtype=np.float32),
            "crop_stage": self.crop_stage,
            "rain": np.array([rain_norm], dtype=np.float32),
            "et0": np.array([et0_norm], dtype=np.float32),
        }
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward signal for the current step.
        
        Reward components:
        1. Water stress penalty: penalizes soil_moisture = 0 (drought)
        2. Irrigation cost penalty: penalizes water use based on action
        
        Parameters
        ----------
        action : int
            Irrigation action {0, 1, 2}
        
        Returns
        -------
        reward : float
            Reward signal for this step
        """
        # Penalty for water stress (soil completely dry)
        stress_penalty = 0.0
        if self.soil_moisture <= 0.0:
            stress_penalty = -1.0
        
        # Penalty for irrigation cost (water usage)
        irrigation_amount = self.irrigation_amounts[action]
        irrigation_cost_penalty = -self.water_cost * irrigation_amount
        
        # Total reward
        reward = stress_penalty + irrigation_cost_penalty
        
        return reward
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        options : dict, optional
            Additional reset options
        
        Returns
        -------
        observation : dict
            Initial observation
        info : dict
            Additional information
        """
        super().reset(seed=seed)
        
        # Initialize state variables
        self.soil_moisture = self.initial_soil_moisture
        self.current_step = 0
        self.crop_stage = self._get_crop_stage(0)
        
        # Sample initial climate
        self.current_et0, self.current_rain = self._sample_climate()
        
        observation = self._get_obs()
        info = {
            "raw_et0": self.current_et0,
            "raw_rain": self.current_rain,
            "soil_moisture_mm": self.soil_moisture * self.max_soil_moisture,
        }
        
        return observation, info
    
    def step(self, action):
        """
        Execute one timestep of the environment.
        
        Parameters
        ----------
        action : int
            Irrigation action {0, 1, 2}
        
        Returns
        -------
        observation : dict
            Updated observation
        reward : float
            Reward signal (placeholder, not implemented yet)
        terminated : bool
            Whether episode has ended
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information
        """
        # Update state based on action and climate
        self._update_state(action)
        
        # Get new observation
        observation = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
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
        }
        
        return observation, reward, terminated, truncated, info


# Example usage and sensitivity analysis
if __name__ == "__main__":
    print("=" * 70)
    print("Irrigation Environment - Climate Configuration Demo")
    print("=" * 70)
    
    # Scenario 1: Arid climate (low rainfall, high ET₀)
    print("\n[Scenario 1] Arid Climate")
    print("-" * 40)
    env_arid = IrrigationEnv(
        max_et0=10.0,
        max_rain=30.0,
        et0_range=(5.0, 10.0),  # High ET₀
        rain_range=(0.0, 5.0),   # Low rainfall
    )
    
    obs, info = env_arid.reset(seed=42)
    print(f"Initial soil moisture: {obs['soil_moisture'][0]:.3f}")
    print(f"ET₀ (raw): {info['raw_et0']:.2f} mm/day")
    print(f"Rain (raw): {info['raw_rain']:.2f} mm/day")
    print(f"Crop stage: {obs['crop_stage']}")
    
    # Scenario 2: Humid climate (high rainfall, moderate ET₀)
    print("\n[Scenario 2] Humid Climate")
    print("-" * 40)
    env_humid = IrrigationEnv(
        max_et0=8.0,
        max_rain=60.0,
        et0_range=(2.0, 5.0),    # Moderate ET₀
        rain_range=(10.0, 50.0),  # High rainfall
    )
    
    obs, info = env_humid.reset(seed=42)
    print(f"Initial soil moisture: {obs['soil_moisture'][0]:.3f}")
    print(f"ET₀ (raw): {info['raw_et0']:.2f} mm/day")
    print(f"Rain (raw): {info['raw_rain']:.2f} mm/day")
    print(f"Crop stage: {obs['crop_stage']}")
    
    # Scenario 3: Mediterranean climate (seasonal variation)
    print("\n[Scenario 3] Mediterranean Climate")
    print("-" * 40)
    env_med = IrrigationEnv(
        max_et0=9.0,
        max_rain=40.0,
        et0_range=(3.0, 8.0),
        rain_range=(0.0, 30.0),
    )
    
    obs, info = env_med.reset(seed=42)
    print(f"Initial soil moisture: {obs['soil_moisture'][0]:.3f}")
    print(f"ET₀ (raw): {info['raw_et0']:.2f} mm/day")
    print(f"Rain (raw): {info['raw_rain']:.2f} mm/day")
    
    # Run a few steps to demonstrate dynamics
    print("\n[Simulation] 5 steps with no irrigation (action=0)")
    print("-" * 40)
    for step in range(5):
        obs, reward, terminated, truncated, info = env_med.step(action=0)
        print(f"Day {info['step']:2d} | "
              f"SM: {obs['soil_moisture'][0]:.3f} | "
              f"ET₀: {info['raw_et0']:.2f} mm | "
              f"Rain: {info['raw_rain']:.2f} mm | "
              f"Stage: {info['crop_stage']}")
    
    print("\n" + "=" * 70)
    print("Environment ready for training and sensitivity analysis!")
    print("=" * 70)
