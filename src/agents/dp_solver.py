import numpy as np
# Import the helpers from your Q-table file
from irr_Qtable import from_discrate_to_full_state, discretize_state, get_state_space_size

def run_value_iteration(env, n_soil_bins, gamma=0.99, theta=1e-6):
    n_stages = 3
    n_states = get_state_space_size(n_soil_bins)
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int) # Initialize policy array
    
    avg_et0 = (env.et0_range[0] + env.et0_range[1]) / 2
    avg_rain = (env.rain_range[0] + env.rain_range[1]) / 2

    # Value Iteration
    while True:
        delta = 0
        for s in range(n_states):
            v_old = V[s]
            soil_idx, stage, _, _ = from_discrate_to_full_state(s, n_soil_bins)
            soil_val = soil_idx / (n_soil_bins - 1) if n_soil_bins > 1 else 0
            
            action_values = []
            for a in range(3):
                irrigation = env.irrigation_amounts[a]
                kc = env.kc_by_stage[stage]
                
                # Physics calculation for next state
                moisture_mm = soil_val * env.max_soil_moisture
                moisture_mm += irrigation + avg_rain - (avg_et0 * kc)
                next_moisture = np.clip(moisture_mm, 0, env.max_soil_moisture) / env.max_soil_moisture
                
                # Set env variables for reward calculation
                env.prev_soil_moisture = soil_val
                env.soil_moisture = next_moisture
                reward = env._calculate_reward(a)
                
                # Predict next state index (Assuming avg weather)
                next_obs = {"soil_moisture": [next_moisture], "crop_stage": stage, "rain": [0.5], "et0": [0.5]}
                next_s = discretize_state(next_obs, n_soil_bins)
                
                action_values.append(reward + gamma * V[next_s])
            
            V[s] = max(action_values)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta: break
            
    # Derive optimal policy for ALL states
    for s in range(n_states):
        soil_idx, stage, _, _ = from_discrate_to_full_state(s, n_soil_bins)
        soil_val = soil_idx / (n_soil_bins - 1) if n_soil_bins > 1 else 0
        
        action_values = []
        for a in range(3):
            irrigation = env.irrigation_amounts[a]
            kc = env.kc_by_stage[stage]
            moisture_mm = soil_val * env.max_soil_moisture
            moisture_mm += irrigation + avg_rain - (avg_et0 * kc)
            next_moisture = np.clip(moisture_mm, 0, env.max_soil_moisture) / env.max_soil_moisture
            
            env.prev_soil_moisture = soil_val
            env.soil_moisture = next_moisture
            reward = env._calculate_reward(a)
            
            next_obs = {"soil_moisture": [next_moisture], "crop_stage": stage, "rain": [0.5], "et0": [0.5]}
            next_s = discretize_state(next_obs, n_soil_bins)
            action_values.append(reward + gamma * V[next_s])
        
        policy[s] = np.argmax(action_values)

    return V, policy
