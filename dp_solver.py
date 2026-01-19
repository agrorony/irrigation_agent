import numpy as np

def run_value_iteration(env, n_soil_bins, gamma=0.99, theta=1e-6):
    """
    Finds the optimal policy using Value Iteration.
    
    :param n_soil_bins: The 'N' size of your problem (Moisture levels) [cite: 14]
    :param gamma: Discount factor [cite: 88]
    :param theta: Convergence threshold
    """
    n_stages = 3  # Emergence, Flowering, Maturity
    n_states = n_soil_bins * n_stages
    n_actions = 3 # 0: None, 1: Light, 2: Heavy
    
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    
    print(f"Starting Value Iteration for N={n_soil_bins}...")

    # 1. Value Iteration Loop
    while True:
        delta = 0
        for s in range(n_states):
            v_old = V[s]
            
            action_values = []
            for a in range(n_actions):
                # We calculate the expected reward for this state-action pair
                # In your env, we can approximate this by looking at the 
                # reward logic you defined [cite: 83, 86]
                
                # Mock moisture and stage from s
                m_idx = s // n_stages
                stage = s % n_stages
                m_val = m_idx / (n_soil_bins - 1)
                
                # Immediate reward based on your specific logic
                reward = env._calculate_reward(a) 
                
                # Transition approximation: 
                # Where would this state go? (Deterministic part of your env)
                if a == 0: next_m = max(0, m_idx - 1)
                elif a == 1: next_m = min(n_soil_bins - 1, m_idx + 1)
                else: next_m = n_soil_bins - 1
                
                next_s = next_m * n_stages + stage # Staying in same stage for simplicity per step
                
                action_values.append(reward + gamma * V[next_s])
            
            V[s] = max(action_values)
            delta = max(delta, abs(v_old - V[s]))
        
        if delta < theta:
            break

    # 2. Policy Extraction
    for s in range(n_states):
        action_values = []
        for a in range(n_actions):
            m_idx = s // n_stages
            m_val = m_idx / (n_soil_bins - 1)
            reward = env._calculate_reward(a)
            
            if a == 0: next_m = max(0, m_idx - 1)
            elif a == 1: next_m = min(n_soil_bins - 1, m_idx + 1)
            else: next_m = n_soil_bins - 1
            
            next_s = next_m * n_stages + (s % n_stages)
            action_values.append(reward + gamma * V[next_s])
            
        policy[s] = np.argmax(action_values)
        
    return V, policy

# Usage Example:
# V_opt, DP_policy = run_value_iteration(env, n_soil_bins=10)
