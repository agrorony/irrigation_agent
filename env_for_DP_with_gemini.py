"""
Irrigation scheduling environment for reinforcement learning.

A Gymnasium-compliant environment for irrigation decision-making with:
- Configurable climatic regimes (ET₀ and rainfall ranges)
- Single-layer soil moisture dynamics
- Discrete crop growth stages
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
    reset_soil_moisture_range : tuple, optional
        (min, max) range for sampling initial soil moisture [0, 1]. If None, uses initial_soil_moisture
    reset_crop_stage_random : bool, optional
        If True, randomly sample initial crop stage from {0, 1, 2}. Default False (starts at 0)
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
        et0_delta_range: tuple = (-2.0, 2.0),
        rain_delta_range: tuple = (-10.0, 10.0),
        threshold_bottom_soil_moisture: float = 0.3,
        threshold_top_soil_moisture: float = 0.7,
        reset_soil_moisture_range: tuple = None,
        reset_crop_stage_random: bool = False
    ):
        super().__init__() 
        
        # Inside IrrigationEnv.__init__
        self.threshold_bottom_soil_moisture = threshold_bottom_soil_moisture
        self.threshold_top_soil_moisture = threshold_top_soil_moisture
        self.prev_soil_moisture = initial_soil_moisture
        
        # Store climate configuration parameters
        self.max_et0 = max_et0  # For normalization (mm/day)
        self.max_rain = max_rain  # For normalization (mm/day)
        self.et0_range = et0_range  # Sampling range (mm/day)
        self.rain_range = rain_range  # Sampling range (mm/day)
        self.et0_delta_range = et0_delta_range  # Daily change range for ET₀ (mm/day)
        self.rain_delta_range = rain_delta_range  # Daily change range for rain (mm/day)
        
        # Soil parameters
        self.max_soil_moisture = max_soil_moisture  # Maximum water capacity (mm)
        self.initial_soil_moisture = initial_soil_moisture  # Initial moisture fraction [0, 1]
        
        # Reset parameters
        self.reset_soil_moisture_range = reset_soil_moisture_range
        self.reset_crop_stage_random = reset_crop_stage_random
        
        # Episode parameters
        self.episode_length = episode_length  # Days per episode
        self.current_step = 0
        
        # Irrigation amounts (mm/day)
        self.irrigation_amounts = np.array([0.0, 5.0, 15.0])
        
        # Crop coefficient by growth stage (for ET calculation)
        # Kc values: emergence (low), flowering (high), maturity (medium)
        self.kc_by_stage = np.array([0.5, 1.15, 0.7])
        
        # Reward function parameters
        self.water_cost = 0.01  # Cost per mm of irrigation applied
        self.threshold_bottom_soil_moisture = threshold_bottom_soil_moisture
        self.threshold_top_soil_moisture = threshold_top_soil_moisture
        
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
    
    def _sample_climate(self, current_et0: float = None, current_rain: float = None) -> tuple:
        """
        Sample climatic conditions using delta changes from previous day.
        If no current values provided, samples from full range (initial conditions).
        
        Parameters
        ----------
        current_et0 : float, optional
            Current ET₀ value (mm/day). If None, samples from full range.
        current_rain : float, optional
            Current rainfall value (mm/day). If None, samples from full range.
        
        Returns
        -------
        et0 : float
            Reference evapotranspiration (mm/day)
        rain : float
            Effective rainfall (mm/day)
        """
        if current_et0 is None or current_rain is None:
            # Initial sampling: use full range
            et0 = np.random.uniform(self.et0_range[0], self.et0_range[1])
            rain = np.random.uniform(self.rain_range[0], self.rain_range[1])
        else:
            # Sample deltas (changes from previous day)
            et0_delta = np.random.uniform(self.et0_delta_range[0], self.et0_delta_range[1])
            rain_delta = np.random.uniform(self.rain_delta_range[0], self.rain_delta_range[1])
            
            # Apply deltas and clip to valid ranges
            et0 = np.clip(current_et0 + et0_delta, self.et0_range[0], self.et0_range[1])
            rain = np.clip(current_rain + rain_delta, self.rain_range[0], self.rain_range[1])
        
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
            Current day of the growing season
        
        Returns
        -------
        stage : int
            Crop growth stage (0=emergence, 1=flowering, 2=maturity)
        """
        # Simplified growth stages:
        # Days 0-30: Emergence (stage 0)
        # Days 31-60: Flowering (stage 1)
        # Days 61+: Maturity (stage 2)
        if day <= 30:
            return 0
        elif day <= 60:
            return 1
        else:
            return 2
    
    def _update_soil_moisture(self, action: int) -> None:
        """
        Update soil moisture based on irrigation action, ET, and rainfall.
        
        Parameters
        ----------
        action : int
            Irrigation action (0=none, 1=light, 2=heavy)
        """
        # Get crop coefficient for current growth stage
        kc = self.kc_by_stage[self.crop_stage]
        
        # Calculate actual ET (ETc = Kc × ET₀)
        etc = kc * self.current_et0  # mm/day
        
        # Get irrigation amount
        irrigation = self.irrigation_amounts[action]  # mm/day
        
        # Calculate net change in soil moisture (mm)
        # Inputs: irrigation + rainfall
        # Outputs: ET consumption
        net_change = irrigation + self.current_rain - etc
        
        # Update soil moisture (as fraction)
        self.soil_moisture += net_change / self.max_soil_moisture
        
        # Clip to valid range [0, 1]
        self.soil_moisture = np.clip(self.soil_moisture, 0.0, 1.0)
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward based on soil moisture state and irrigation cost.
        
        Reward structure:
        1. Penalty for water stress (very dry soil)
        2. Penalty for over-irrigation (waterlogged soil)
        3. Bonus for maintaining optimal moisture
        4. Cost for irrigation water usage
        
        Parameters
        ----------
        action : int
            Irrigation action taken
        
        Returns
        -------
        reward : float
            Reward value for this time step
        """
        reward = 0 
        prev = self.prev_soil_moisture
        curr = self.soil_moisture
        bottom = self.threshold_bottom_soil_moisture
        top = self.threshold_top_soil_moisture

        # entering optimum 
        if prev < bottom and bottom <= curr <= top:
            reward += 6

        # staying in optimal 
        if bottom <= prev <= top and bottom <= curr <= top:
            reward += 2

        # Leaving optimal zone
        if bottom <= prev <= top and (curr < bottom or curr > top):
            reward -= 4.0

        # ---------- 2. Continuous stress penalties ----------
        # Below optimal
        if curr < bottom:
            reward -= 50.0 * (bottom - curr)

        # Above optimal
        if curr > top:
            reward -= 1.0 * (curr - top)

        # ---------- 3. Irrigation cost ----------
        irrigation_amount = self.irrigation_amounts[action]
        reward -= self.water_cost * irrigation_amount
    
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
        
        # Initialize soil moisture
        if self.reset_soil_moisture_range is not None:
            # Sample from range
            self.soil_moisture = np.random.uniform(
                self.reset_soil_moisture_range[0], 
                self.reset_soil_moisture_range[1]
            )
        else:
            # Use fixed initial value
            self.soil_moisture = self.initial_soil_moisture
        
        # Initialize crop stage
        if self.reset_crop_stage_random:
            self.crop_stage = np.random.randint(0, 3)
        else:
            self.crop_stage = 0  # Start at emergence
        
        self.current_step = 0
        
        # Sample initial climate conditions
        self.current_et0, self.current_rain = self._sample_climate()
        
        # Get initial observation
        observation = self._get_obs()
        info = {}
        self.prev_soil_moisture = self.soil_moisture
        
        return observation, info
    
    def step(self, action):
        """
        Execute one time step in the environment.
        
        Parameters
        ----------
        action : int
            Action to take (0=no irrigation, 1=light, 2=heavy)
        
        Returns
        -------
        observation : dict
            Current observation
        reward : float
            Reward for this step
        terminated : bool
            Whether episode has ended
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Update step counter
        self.current_step += 1
        
        # Update crop growth stage
        self.crop_stage = self._get_crop_stage(self.current_step)
        
        # Sample new climate conditions (with temporal correlation)
        self.current_et0, self.current_rain = self._sample_climate(
            self.current_et0, self.current_rain
        )
        
        # Update soil moisture based on action, ET, and rainfall
        self._update_soil_moisture(action)
        
        # Get observation
        observation = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check if episode is complete
        terminated = self.current_step >= self.episode_length
        truncated = False  # No truncation conditions
        
        # Additional info
        info = {
            "soil_moisture": self.soil_moisture,
            "crop_stage": self.crop_stage,
            "et0": self.current_et0,
            "rain": self.current_rain,
            "irrigation": self.irrigation_amounts[action],
        }
        
        self.prev_soil_moisture = self.soil_moisture
        
        return observation, reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Get current observation.
        
        Returns
        -------
        observation : dict
            Current state observation
        """
        # Normalize climate variables
        et0_norm, rain_norm = self._normalize_climate(self.current_et0, self.current_rain)
        
        return {
            "soil_moisture": np.array([self.soil_moisture], dtype=np.float32),
            "crop_stage": self.crop_stage,
            "rain": np.array([rain_norm], dtype=np.float32),
            "et0": np.array([et0_norm], dtype=np.float32),
        }
    
    def render(self):
        """Render environment (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
