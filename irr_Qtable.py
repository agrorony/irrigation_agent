"""
Tabular Q-learning for Irrigation Scheduling Environment

Implements discrete state-action Q-learning without modifying the environment.
"""

import numpy as np
from irrigation_env import IrrigationEnv


# ============================================================================
# 1. STATE DISCRETIZATION
# ============================================================================

def discretize_state(observation, n_soil_bins=12):
    """
    Convert continuous observation to discrete state index.
    
    Parameters
    ----------
    observation : dict
        Environment observation with keys: soil_moisture, crop_stage, rain, et0
    n_soil_bins : int
        Number of bins for soil moisture [0, 1], default 12 for fewer states
    
    Returns
    -------
    state_index : int
        Discrete state index
    """
    # Extract and clip values to [0, 1]
    soil_moisture = np.clip(observation['soil_moisture'][0], 0.0, 1.0)
    crop_stage = observation['crop_stage']
    rain = np.clip(observation['rain'] / 50.0, 0.0, 1.0)  # Assuming max rain = 50 mm
    et0 = np.clip(observation['et0'] / 8.0, 0.0, 1.0)  # Assuming max et0 = 8 mm/day
    
    # Discretize soil moisture into fewer bins 
    soil_bin = int(soil_moisture * n_soil_bins)
    if soil_bin >= n_soil_bins:
        soil_bin = n_soil_bins - 1

    # Binary ET₀: 0 = low, 1 = high (threshold at 0.5)
    et0_bin = 1 if et0 >= 0.5 else 0
    
    # Binary rain: 0 = no rain, 1 = rain (threshold at 0.1)
    rain_bin = 1 if rain >= 0.1 else 0
    
    # Combine into single state index
    # State = (soil_bin, crop_stage, et0_bin, rain_bin)
    # ET₀ and rain are now binary (2 states each)
    state_index = (
        soil_bin * (3 * 2 * 2) +
        crop_stage * (2 * 2) +
        et0_bin * 2 +
        rain_bin
    )
    
    return state_index


def get_state_space_size(n_soil_bins=12, n_crop_stages=3):
    """
    Calculate total number of discrete states.
    
    Returns
    -------
    n_states : int
        Total number of discrete states (simplified: soil_bins × crop_stages)
    """
    # Simplified state space: only soil moisture × crop stages
    return n_soil_bins * n_crop_stages*2*2


# ============================================================================
# 2. DISCRETE ACTION SPACE
# ============================================================================

# Action indices and their corresponding irrigation depths (mm)
ACTION_SPACE = {
    0: 0.0,   # No irrigation
    1: 5.0,   # Light irrigation
    2: 15.0,  # Heavy irrigation
}

N_ACTIONS = len(ACTION_SPACE)


# ============================================================================
# 3. Q-TABLE INITIALIZATION
# ============================================================================

def initialize_q_table(n_states, n_actions):
    """
    Initialize Q-table with zeros.
    
    Parameters
    ----------
    n_states : int
        Number of discrete states
    n_actions : int
        Number of discrete actions
    
    Returns
    -------
    Q : np.ndarray
        Q-table of shape (n_states, n_actions)
    """
    return np.zeros((n_states, n_actions))


# ============================================================================
# 4. Q-LEARNING ALGORITHM
# ============================================================================

def epsilon_greedy_action(Q, state_index, epsilon, n_actions):
    """
    Select action using epsilon-greedy policy.
    
    Parameters
    ----------
    Q : np.ndarray
        Q-table
    state_index : int
        Current discrete state
    epsilon : float
        Exploration probability
    n_actions : int
        Number of actions
    
    Returns
    -------
    action : int
        Selected action index
    """
    if np.random.random() < epsilon:
        # Explore: random action
        return np.random.randint(n_actions)
    else:
        # Exploit: best action
        return np.argmax(Q[state_index])


def q_learning_update(Q, state, action, reward, next_state, done, alpha, gamma):
    """
    Update Q-table using Q-learning rule.
    
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Parameters
    ----------
    Q : np.ndarray
        Q-table
    state : int
        Current state index
    action : int
        Action taken
    reward : float
        Reward received
    next_state : int
        Next state index
    done : bool
        Whether episode terminated
    alpha : float
        Learning rate
    gamma : float
        Discount factor
    """
    if done:
        # No future rewards if episode is done
        target = reward
    else:
        # Bootstrap from next state
        target = reward + gamma * np.max(Q[next_state])
    
    # Q-learning update
    Q[state, action] += alpha * (target - Q[state, action])


def train_q_learning(
    env,
    n_episodes=1000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    n_soil_bins=12,
    Q_init=None,
):
    """
    Train Q-learning agent on irrigation environment.
    
    Parameters
    ----------
    env : IrrigationEnv
        Irrigation environment instance
    n_episodes : int
        Number of training episodes
    alpha : float
        Learning rate
    gamma : float
        Discount factor
    epsilon_start : float
        Initial exploration rate
    epsilon_end : float
        Final exploration rate
    epsilon_decay : float
        Epsilon decay rate per episode
    n_soil_bins : int
        Number of bins for soil moisture discretization
    Q_init : np.ndarray, optional
        Initial Q-table to continue training from. If None, initializes new table.
    
    Returns
    -------
    Q : np.ndarray
        Trained Q-table
    """
    # Initialize or use provided Q-table
    if Q_init is None:
        n_states = get_state_space_size(n_soil_bins)
        Q = initialize_q_table(n_states, N_ACTIONS)
    else:
        Q = Q_init.copy()
    
    # Epsilon for exploration
    epsilon = epsilon_start
    
    # Training loop
    for episode in range(n_episodes):
        # Reset environment
        observation, info = env.reset()
        state = discretize_state(observation, n_soil_bins)
        done = False
        
        # Episode loop
        while not done:
            # Select action using epsilon-greedy
            action_index = epsilon_greedy_action(Q, state, epsilon, N_ACTIONS)
            
            # Execute action in environment
            observation, reward, terminated, truncated, info = env.step(action_index)
            done = terminated or truncated
            
            # Discretize next state
            next_state = discretize_state(observation, n_soil_bins)
            
            # Update Q-table
            q_learning_update(Q, state, action_index, reward, next_state, done, alpha, gamma)
            
            # Move to next state
            state = next_state
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    return Q


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create environment
    env = IrrigationEnv(
        max_et0=8.0,
        max_rain=50.0,
        et0_range=(2.0, 8.0),
        rain_range=(0.0, 40.0),
        episode_length=90,
    )
    
    # Train Q-learning agent
    Q = train_q_learning(
        env=env,
        n_episodes=1000,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        n_soil_bins=12,
    )
    
    print("Q-learning training complete")
    print(f"Q-table shape: {Q.shape}")
    print(f"Non-zero entries: {np.count_nonzero(Q)}/{Q.size}")
