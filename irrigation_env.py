"""
Irrigation Scheduling Environment

A Gymnasium-compliant environment for irrigation scheduling optimization
using reinforcement learning.

Features:
- Configurable climatic regimes (ET₀ and rainfall ranges)
- Single-layer soil moisture dynamics
- Discrete crop growth stages
- Full Gymnasium API compliance
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
        - 1: Light irrigation (5 mm/day)
        - 2: Heavy irrigation (15 mm/day)
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
        threshold_top_soil_moisture: float = 0.7
    ):
        """
        Initialize the irrigation environment.
        
        Parameters
        ----------
        max_et0 : float
            Maximum ET₀ for normalization (mm/day)
        max_rain : float
            Maximum rainfall for normalization (mm/day)
        et0_range : tuple
            ET₀ sampling range (min, max) in mm/day
        rain_range : tuple
            Rainfall sampling range (min, max) in mm/day
        max_soil_moisture : float
            Maximum water holding capacity (mm)
        initial_soil_moisture : float
            Initial soil moisture as fraction [0, 1]
        episode_length : int
            Number of days per episode
        et0_delta_range : tuple
            Daily ET₀ change range (min, max) in mm/day
        rain_delta_range : tuple
            Daily rainfall change range (min, max) in mm/day
        threshold_bottom_soil_moisture : float
            Lower threshold for water stress penalty
        threshold_top_soil_moisture : float
            Upper threshold for excess water penalty
        """
        super().__init__()
        
        # Store climate configuration parameters
        self.max_et0 = max_et0
        self.max_rain = max_rain
        self.et0_range = et0_range
        self.rain_range = rain_range
        self.et0_delta_range = et0_delta_range
        self.rain_delta_range = rain_delta_range
        
        # Soil parameters
        self.max_soil_moisture = max_soil_moisture
        self.initial_soil_moisture = initial_soil_moisture
        
        # Episode parameters
        self.episode_length = episode_length
        self.current_step = 0
        
        # Irrigation amounts (mm/day)
        self.irrigation_amounts = np.array([0.0, 5.0, 15.0])
        
        # Crop coefficient by growth stage
        self.kc_by_stage = np.array([0.5, 1.15, 0.7])
        
        # Reward function parameters
        self.water_cost = 0.1
        self.threshold_bottom_soil_moisture = threshold_bottom_soil_moisture
        self.threshold_top_soil_moisture = threshold_top_soil_moisture
        
        # Define observation space
        self.observation_space = spaces.Dict({
            "soil_moisture": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "crop_stage": spaces.Discrete(3),
            "rain": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            "et0": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        })
        
        # Define action space
        self.action_space = spaces.Discrete(3)
        
        # Current state variables
        self.soil_moisture = None
        self.crop_stage = None
        self.current_rain = None
        self.current_et0 = None
    
    def _sample_climate(self, current_et0: float = None, current_rain: float = None) -> tuple:
        """
        Sample climatic conditions using delta changes from previous day.
        
        Parameters
        ----------
        current_et0 : float, optional
            Current ET₀ value (mm/day)
        current_rain : float, optional
            Current rainfall value (mm/day)
        
        Returns
        -------
        et0 : float
            Sampled ET₀ value (mm/day)
        rain : float
            Sampled rainfall value (mm/day)
        """
        if current_et0 is None or current_rain is None:
            et0 = np.random.uniform(self.et0_range[0], self.et0_range[1])
            rain = np.random.uniform(self.rain_range[0], self.rain_range[1])
        else:
            et0_delta = np.random.uniform(self.et0_delta_range[0], self.et0_delta_range[1])
            rain_delta = np.random.uniform(self.rain_delta_range[0], self.rain_delta_range[1])
            et0 = np.clip(current_et0 + et0_delta, self.et0_range[0], self.et0_range[1])
            rain = np.clip(current_rain + rain_delta, self.rain_range[0], self.rain_range[1])
        
        return et0, rain
    
    def _normalize_climate(self, et0: float, rain: float) -> tuple:
        """
        Normalize climate variables to [0, 1].
        
        Parameters
        ----------
        et0 : float
            ET₀ value (mm/day)
        rain : float
            Rainfall value (mm/day)
        
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
            Day number in current episode
        
        Returns
        -------
        stage : int
            Crop growth stage (0=emergence, 1=flowering, 2=maturity)
        """
        stage_duration = self.episode_length // 3
        if day < stage_duration:
            return 0
        elif day < 2 * stage_duration:
            return 1
        else:
            return 2
    
    def _update_state(self, action: int):
        """
        Update soil moisture based on action and climate.
        
        Parameters
        ----------
        action : int
            Irrigation action index
        """
        irrigation = self.irrigation_amounts[action]
        kc = self.kc_by_stage[self.crop_stage]
        et_crop = self.current_et0 * kc
        
        moisture_mm = self.soil_moisture * self.max_soil_moisture
        moisture_mm += irrigation + self.current_rain - et_crop
        moisture_mm = np.clip(moisture_mm, 0.0, self.max_soil_moisture)
        
        self.soil_moisture = moisture_mm / self.max_soil_moisture
        self.current_et0, self.current_rain = self._sample_climate(self.current_et0, self.current_rain)
        
        self.current_step += 1
        self.crop_stage = self._get_crop_stage(self.current_step)
    
    def _get_obs(self) -> dict:
        """
        Get current observation.
        
        Returns
        -------
        observation : dict
            Current environment observation
        """
        et0_norm, rain_norm = self._normalize_climate(self.current_et0, self.current_rain)
        
        return {
            "soil_moisture": np.array([self.soil_moisture], dtype=np.float32),
            "crop_stage": self.crop_stage,
            "rain": np.array([rain_norm], dtype=np.float32),
            "et0": np.array([et0_norm], dtype=np.float32),
        }
    
    def _calculate_reward(self, action: int) -> float:
        """
        Calculate reward signal.
        
        Parameters
        ----------
        action : int
            Irrigation action index
        
        Returns
        -------
        reward : float
            Reward value
        """
        stress_penalty = 0.0
        if self.soil_moisture <= self.threshold_bottom_soil_moisture:
            stress_penalty = -1.0 * (self.threshold_bottom_soil_moisture - self.soil_moisture)
        if self.soil_moisture >= self.threshold_top_soil_moisture:
            stress_penalty += -0.5 * (self.soil_moisture - self.threshold_top_soil_moisture)
        
        irrigation_amount = self.irrigation_amounts[action]
        irrigation_cost_penalty = -self.water_cost * irrigation_amount
        
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
        
        self.soil_moisture = self.initial_soil_moisture
        self.current_step = 0
        self.crop_stage = self._get_crop_stage(0)
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
            Irrigation action index
        
        Returns
        -------
        observation : dict
            Environment observation
        reward : float
            Reward value
        terminated : bool
            Whether episode has ended
        truncated : bool
            Whether episode was truncated
        info : dict
            Additional information
        """
        self._update_state(action)
        observation = self._get_obs()
        reward = self._calculate_reward(action)
        terminated = self.current_step >= self.episode_length
        truncated = False
        
        info = {
            "step": self.current_step,
            "raw_et0": self.current_et0,
            "raw_rain": self.current_rain,
            "soil_moisture_mm": self.soil_moisture * self.max_soil_moisture,
            "crop_stage": self.crop_stage,
        }
        
        return observation, reward, terminated, truncated, info
