"""
Deep Q-Network (DQN) for Irrigation Scheduling Environment

Implements standard DQN as taught in class: 
- Neural network for Q-value approximation
- Experience replay buffer
- Target network with periodic updates
- Epsilon-greedy exploration

Public API:
    DQN              - Neural network class
    ReplayBuffer     - Experience replay buffer
    DQNAgent         - DQN agent with training logic
    train_dqn()      - Train a DQN agent
    evaluate_policy() - Evaluate trained agent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from pathlib import Path


# ============================================================================
# 1. NEURAL NETWORK ARCHITECTURE
# ============================================================================

class DQN(nn.Module):
    """
    Deep Q-Network with fully connected layers. 
    
    Architecture:
        Input → FC(64) → ReLU → FC(64) → ReLU → FC(n_actions)
    """
    
    def __init__(self, state_dim, n_actions, hidden_size=64):
        """
        Initialize DQN. 
        
        Parameters
        ----------
        state_dim : int
            Dimension of state vector
        n_actions : int
            Number of actions
        hidden_size : int
            Size of hidden layers
        """
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, n_actions)
        
    def forward(self, x):
        """
        Forward pass. 
        
        Parameters
        ----------
        x : torch.Tensor
            State tensor of shape (batch_size, state_dim)
        
        Returns
        -------
        q_values : torch. Tensor
            Q-values of shape (batch_size, n_actions)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self. fc3(x)


# ============================================================================
# 2. REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for storing transitions.
    """
    
    def __init__(self, capacity):
        """
        Initialize replay buffer.
        
        Parameters
        ----------
        capacity : int
            Maximum buffer size
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add transition to buffer.
        
        Parameters
        ----------
        state : np.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Next state
        done : bool
            Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample random batch of transitions.
        
        Parameters
        ----------
        batch_size : int
            Number of transitions to sample
        
        Returns
        -------
        batch : tuple
            (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Return current buffer size."""
        return len(self.buffer)


# ============================================================================
# 3. STATE CONVERSION
# ============================================================================

def observation_to_state(observation):
    """
    Convert environment observation dict to state vector.
    
    Parameters
    ----------
    observation :  dict
        Environment observation with keys:  soil_moisture, crop_stage, rain, et0
    
    Returns
    -------
    state : np.ndarray
        Flattened state vector [soil_moisture, crop_stage, rain, et0]
    """
    return np.array([
        observation['soil_moisture'][0],
        float(observation['crop_stage']),
        observation['rain'][0],
        observation['et0'][0]
    ], dtype=np.float32)


# ============================================================================
# 4. DQN AGENT
# ============================================================================

class DQNAgent: 
    """
    DQN Agent with training and action selection methods.
    """
    
    def __init__(
        self,
        state_dim,
        n_actions,
        learning_rate=0.001,
        gamma=0.99,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        hidden_size=64,
        device=None
    ):
        """
        Initialize DQN agent.
        
        Parameters
        ----------
        state_dim : int
            Dimension of state vector
        n_actions : int
            Number of actions
        learning_rate : float
            Learning rate for optimizer
        gamma : float
            Discount factor
        buffer_capacity :  int
            Replay buffer capacity
        batch_size : int
            Minibatch size for training
        target_update_freq : int
            Frequency of target network updates (in steps)
        hidden_size :  int
            Size of hidden layers
        device : torch.device, optional
            Device to use for training
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self. device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize networks
        self.q_network = DQN(state_dim, n_actions, hidden_size).to(self.device)
        self.target_network = DQN(state_dim, n_actions, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training step counter
        self.steps = 0
        
        # Training statistics
        self.total_episodes = 0
    
    def save(self, filepath):
        """
        Save DQN agent state to file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save checkpoint
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'total_episodes': self.total_episodes,
            'state_dim': self.state_dim,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load DQN agent state from file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to load checkpoint from
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.total_episodes = checkpoint.get('total_episodes', 0)
        
        print(f"Model loaded from {filepath}")
        print(f"Resuming from episode {self.total_episodes}, step {self.steps}")
    
    def select_action(self, state, epsilon):
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        state : np.ndarray
            Current state vector
        epsilon : float
            Exploration probability
        
        Returns
        -------
        action : int
            Selected action
        """
        if random.random() < epsilon:
            # Explore: random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def train_step(self):
        """
        Perform one training step (sample batch and update Q-network).
        
        Returns
        -------
        loss :  float
            Training loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize
        self. optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self. target_network.load_state_dict(self.q_network. state_dict())
        
        return loss.item()


# ============================================================================
# 5. TRAINING FUNCTION
# ============================================================================

def train_dqn(
    env,
    n_episodes=1000,
    epsilon_start=1.0,
    epsilon_end=0.1,
    epsilon_decay=0.995,
    learning_rate=0.001,
    gamma=0.99,
    buffer_capacity=10000,
    batch_size=64,
    target_update_freq=100,
    hidden_size=64,
    verbose=True,
    save_dir="models/dqn",
    checkpoint_freq=500,
    load_checkpoint=None
):
    """
    Train DQN agent on irrigation environment.
    
    Parameters
    ----------
    env : IrrigationEnv
        Irrigation environment instance
    n_episodes : int
        Number of training episodes
    epsilon_start : float
        Initial exploration rate
    epsilon_end : float
        Final exploration rate
    epsilon_decay :  float
        Epsilon decay rate per episode
    learning_rate : float
        Learning rate for optimizer
    gamma : float
        Discount factor
    buffer_capacity : int
        Replay buffer capacity
    batch_size : int
        Minibatch size
    target_update_freq : int
        Target network update frequency (steps)
    hidden_size : int
        Size of hidden layers
    verbose : bool
        Whether to print training progress
    save_dir : str
        Directory to save model checkpoints
    checkpoint_freq : int
        Save checkpoint every N episodes
    load_checkpoint : str, optional
        Path to checkpoint to resume training from
    
    Returns
    -------
    agent : DQNAgent
        Trained DQN agent
    episode_returns : list
        List of episode returns during training
    """
    # Infer state and action dimensions
    state_dim = 4  # [soil_moisture, crop_stage, rain, et0]
    n_actions = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        hidden_size=hidden_size
    )
    
    # Load checkpoint if specified
    if load_checkpoint is not None:
        agent.load(load_checkpoint)
        start_episode = agent.total_episodes
        # Adjust epsilon based on episodes already trained
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** start_episode))
        if verbose:
            print(f"Resuming training from episode {start_episode}")
            print(f"Adjusted epsilon: {epsilon:.3f}\n")
    else:
        start_episode = 0
    
    # Training loop
    epsilon = epsilon if load_checkpoint else epsilon_start
    episode_returns = []
    
    for episode in range(n_episodes):
        observation, _ = env.reset()
        state = observation_to_state(observation)
        done = False
        episode_return = 0
        
        while not done: 
            # Select action
            action = agent.select_action(state, epsilon)
            
            # Execute action
            next_observation, reward, terminated, truncated, _ = env. step(action)
            done = terminated or truncated
            next_state = observation_to_state(next_observation)
            
            # Store transition
            agent. replay_buffer.push(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            
            # Update state
            state = next_state
            episode_return += reward
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Record episode return
        episode_returns. append(episode_return)
        agent.total_episodes = start_episode + episode + 1
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_return = np.mean(episode_returns[-100:])
            print(f"Episode {agent.total_episodes}/{start_episode + n_episodes} | "
                  f"Avg Return: {avg_return:.2f} | "
                  f"Epsilon: {epsilon:.3f}")
        
        # Save checkpoint
        if checkpoint_freq > 0 and (episode + 1) % checkpoint_freq == 0:
            checkpoint_path = Path(save_dir) / f"checkpoint_ep{agent.total_episodes}.pt"
            agent.save(checkpoint_path)
            if verbose:
                print(f"  → Checkpoint saved at episode {agent.total_episodes}")
    
    # Save final model
    final_path = Path(save_dir) / "dqn_final.pt"
    agent.save(final_path)
    
    if verbose:
        print("\nTraining complete!")
        print(f"Final average return (last 100 episodes): {np.mean(episode_returns[-100:]):.2f}")
        print(f"Final model saved to {final_path}")
    
    return agent, episode_returns


# ============================================================================
# 6. EVALUATION FUNCTION
# ============================================================================

def evaluate_policy(agent, env, n_episodes=50, verbose=True):
    """
    Evaluate trained DQN agent with greedy policy (epsilon=0).
    
    Parameters
    ----------
    agent : DQNAgent
        Trained DQN agent
    env : IrrigationEnv
        Irrigation environment
    n_episodes : int
        Number of evaluation episodes
    verbose : bool
        Whether to print results
    
    Returns
    -------
    avg_return : float
        Average return over evaluation episodes
    returns : list
        List of returns for each episode
    """
    returns = []
    
    for episode in range(n_episodes):
        observation, _ = env.reset()
        state = observation_to_state(observation)
        done = False
        episode_return = 0
        
        while not done:
            # Greedy action selection (epsilon=0)
            action = agent.select_action(state, epsilon=0.0)
            
            # Execute action
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = observation_to_state(next_observation)
            
            episode_return += reward
        
        returns.append(episode_return)
    
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    if verbose: 
        print(f"\nEvaluation over {n_episodes} episodes:")
        print(f"  Average Return: {avg_return:.2f} ± {std_return:.2f}")
        print(f"  Min Return: {np.min(returns):.2f}")
        print(f"  Max Return: {np.max(returns):.2f}")
    
    return avg_return, returns


def run_policy_rollout(agent, env, seed=None, max_steps=None, verbose=True):
    """
    Run a single episode with greedy policy and log detailed step information.
    
    Useful for debugging and analyzing trained agent behavior.
    
    Parameters
    ----------
    agent : DQNAgent
        Trained DQN agent
    env : IrrigationEnv
        Environment instance
    seed : int, optional
        Random seed for reproducibility
    max_steps : int, optional
        Maximum steps to run (if None, runs until episode ends)
    verbose : bool
        If True, prints formatted table to console
    
    Returns
    -------
    rollout_data : list of dict
        List of step dictionaries with keys: day, soil_moisture, crop_stage,
        et0, rain, action, irrigation, reward
    """
    # Reset environment
    if seed is not None:
        observation, _ = env.reset(seed=seed)
    else:
        observation, _ = env.reset()
    
    state = observation_to_state(observation)
    done = False
    step_count = 0
    rollout_data = []
    
    # Print header
    if verbose:
        print("\n" + "="*90)
        print("POLICY ROLLOUT - Greedy Action Selection (epsilon=0)")
        print("="*90)
        print(f"{'Day':>4} {'Soil':>8} {'Stage':>6} {'ET0':>8} {'Rain':>8} {'Action':>7} {'Irrig':>8} {'Reward':>8}")
        print(f"{'':>4} {'(frac)':>8} {'':>6} {'(mm/d)':>8} {'(mm/d)':>8} {'':>7} {'(mm)':>8} {'':>8}")
        print("-"*90)
    
    # Run episode
    while not done:
        # Select greedy action
        action = agent.select_action(state, epsilon=0.0)
        
        # Execute action
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Extract raw values from info dict
        soil_moisture = info['soil_moisture']
        crop_stage = info['crop_stage']
        et0 = info['et0']
        rain = info['rain']
        irrigation = info['irrigation']
        
        # Store step data
        step_data = {
            'day': step_count + 1,
            'soil_moisture': soil_moisture,
            'crop_stage': crop_stage,
            'et0': et0,
            'rain': rain,
            'action': action,
            'irrigation': irrigation,
            'reward': reward
        }
        rollout_data.append(step_data)
        
        # Print step
        if verbose:
            print(f"{step_data['day']:4d} {step_data['soil_moisture']:8.3f} "
                  f"{step_data['crop_stage']:6d} {step_data['et0']:8.2f} "
                  f"{step_data['rain']:8.2f} {step_data['action']:7d} "
                  f"{step_data['irrigation']:8.2f} {step_data['reward']:8.2f}")
        
        # Update state
        state = observation_to_state(next_observation)
        step_count += 1
        
        # Check max_steps limit
        if max_steps is not None and step_count >= max_steps:
            break
    
    if verbose:
        print("-"*90)
        total_reward = sum(d['reward'] for d in rollout_data)
        total_irrigation = sum(d['irrigation'] for d in rollout_data)
        print(f"Episode complete: {len(rollout_data)} days")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Total irrigation: {total_irrigation:.2f} mm")
        print("="*90 + "\n")
    
    return rollout_data


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__": 
    from irrigation_agent.irrigation_env import IrrigationEnv
    
    # Create environment
    env = IrrigationEnv(
        max_et0=50.0,
        max_rain=20.0,
        et0_range=(2.0, 50.0),
        rain_range=(0.0, 20.0),
        episode_length=90,
    )
    
    print("Training DQN agent...")
    print(f"State dimension: 4")
    print(f"Action space: {env.action_space. n}")
    print()
    
    # Check for existing checkpoint to resume from
    save_dir = "models/dqn"
    latest_checkpoint = None
    checkpoint_dir = Path(save_dir)
    if checkpoint_dir.exists():
        # Look for final model first
        final_model = checkpoint_dir / "dqn_final.pt"
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_ep*.pt"))
        
        if final_model.exists():
            checkpoints.append(final_model)
        
        if checkpoints:
            # Get the most recent
            latest_checkpoint = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
            print(f"Found checkpoint: {latest_checkpoint}")
            response = input("Resume from this checkpoint? (y/n): ").strip().lower()
            if response != 'y':
                latest_checkpoint = None
                print("Starting fresh training...\n")
            else:
                print("Continuing training from checkpoint...\n")
        else:
            print("No checkpoints found. Starting fresh training...\n")
    else:
        print("No checkpoint directory found. Starting fresh training...\n")
    
    # Train agent
    agent, episode_returns = train_dqn(
        env=env,
        n_episodes=10000,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.999,
        learning_rate=0.001,
        gamma=0.99,
        verbose=True,
        save_dir=save_dir,
        checkpoint_freq=500,
        load_checkpoint=latest_checkpoint
    )
    
    # Evaluate agent
    print("\n" + "="*60)
    print("EVALUATING TRAINED AGENT")
    print("="*60)
    avg_return, returns = evaluate_policy(agent, env, n_episodes=50)
    
    # Run policy rollout for detailed inspection
    print("\n" + "="*60)
    print("POLICY ROLLOUT - Detailed Episode Trace")
    print("="*60)
    rollout_data = run_policy_rollout(agent, env, seed=42)
"""
DQN Agent for Irrigation Scheduling
====================================

Vanilla Deep Q-Network implementation in PyTorch for discrete irrigation control.

Components:
- QNetwork: MLP that maps observations to Q-values for 16 actions
- ReplayBuffer: Experience replay for off-policy learning
- DQNAgent: Main agent with training and action selection

Architecture:
    Input: Dict observation -> Flattened state (6 dims)
        - soil_moisture: 1 dim
        - crop_stage: 3 dims (one-hot encoded)
        - rain: 1 dim
        - et0: 1 dim
    Hidden: [128, 128] with ReLU
    Output: 16 Q-values (one per action)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from pathlib import Path


class QNetwork(nn.Module):
    """
    Q-Network for DQN agent.
    
    Maps flattened observation (6 dims) to Q-values for 16 actions.
    
    Parameters
    ----------
    state_dim : int
        Dimension of flattened state (default: 6)
    action_dim : int
        Number of actions (default: 16)
    hidden_dims : list
        Hidden layer sizes (default: [128, 128])
    """
    
    def __init__(self, state_dim=6, action_dim=16, hidden_dims=[128, 128]):
        super(QNetwork, self).__init__()
        
        # Build MLP layers
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass through network.
        
        Parameters
        ----------
        state : torch.Tensor
            Flattened state tensor [batch_size, state_dim]
        
        Returns
        -------
        q_values : torch.Tensor
            Q-values for each action [batch_size, action_dim]
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    
    Stores transitions (s, a, r, s', done) and samples random batches.
    
    Parameters
    ----------
    capacity : int
        Maximum buffer size (default: 100000)
    """
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add transition to buffer.
        
        Parameters
        ----------
        state : np.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Next state
        done : bool
            Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample random batch of transitions.
        
        Parameters
        ----------
        batch_size : int
            Number of transitions to sample
        
        Returns
        -------
        states, actions, rewards, next_states, dones : tuple of np.ndarray
            Batch of transitions
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for irrigation scheduling.
    
    Parameters
    ----------
    state_dim : int
        Dimension of flattened state (default: 6)
    action_dim : int
        Number of discrete actions (default: 16)
    learning_rate : float
        Learning rate for optimizer (default: 1e-4)
    gamma : float
        Discount factor (default: 0.99)
    epsilon_start : float
        Initial exploration rate (default: 1.0)
    epsilon_end : float
        Final exploration rate (default: 0.05)
    epsilon_decay_steps : int
        Steps over which to decay epsilon (default: 50000)
    buffer_size : int
        Replay buffer capacity (default: 100000)
    batch_size : int
        Minibatch size for training (default: 64)
    target_update_freq : int
        Frequency of target network updates (default: 1000)
    device : str
        Device for training ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        state_dim=6,
        action_dim=16,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.training_steps = 0
        
        # Networks
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.update_target_network()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    def obs_to_state(self, obs):
        """
        Convert Dict observation to flat numpy array.
        
        One-hot encodes crop_stage and concatenates with other features.
        
        Parameters
        ----------
        obs : dict
            Dictionary observation with keys:
                - 'soil_moisture': [1] array
                - 'crop_stage': int (0, 1, or 2)
                - 'rain': [1] array
                - 'et0': [1] array
        
        Returns
        -------
        state : np.ndarray
            Flattened state array of shape (6,)
        """
        # Extract components
        soil_moisture = float(obs['soil_moisture'][0])
        crop_stage = int(obs['crop_stage'])
        rain = float(obs['rain'][0])
        et0 = float(obs['et0'][0])
        
        # One-hot encode crop stage (3 stages: 0, 1, 2)
        crop_stage_onehot = np.zeros(3, dtype=np.float32)
        crop_stage_onehot[crop_stage] = 1.0
        
        # Concatenate into single state vector
        state = np.array([
            soil_moisture,
            crop_stage_onehot[0],
            crop_stage_onehot[1],
            crop_stage_onehot[2],
            rain,
            et0
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, obs, evaluate=False):
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        obs : dict
            Dictionary observation
        evaluate : bool
            If True, use greedy policy (no exploration)
        
        Returns
        -------
        action : int
            Selected action index
        """
        # Convert observation to state
        state = self.obs_to_state(obs)
        
        # Epsilon-greedy action selection
        if not evaluate and random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action
    
    def train_step(self):
        """
        Perform one training step.
        
        Samples batch from replay buffer, computes TD target, and updates Q-network.
        
        Returns
        -------
        loss : float
            Training loss value, or None if buffer has insufficient samples
        """
        # Check if buffer has enough samples
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.training_steps += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
            self.training_steps / self.epsilon_decay_steps
        )
        
        # Update target network
        if self.training_steps % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Hard update of target network (copy weights from Q-network)."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, path):
        """
        Save model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to save checkpoint
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay_steps': self.epsilon_decay_steps,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path):
        """
        Load model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']


# Verification and testing
if __name__ == "__main__":
    print("=" * 70)
    print("DQN Agent - Component Test")
    print("=" * 70)
    
    # Test 1: QNetwork
    print("\n1. Testing QNetwork...")
    q_net = QNetwork(state_dim=6, action_dim=16, hidden_dims=[128, 128])
    dummy_state = torch.randn(1, 6)
    q_values = q_net(dummy_state)
    print(f"   Input shape: {dummy_state.shape}")
    print(f"   Output shape: {q_values.shape}")
    print(f"   ✓ QNetwork forward pass successful")
    
    # Test 2: ReplayBuffer
    print("\n2. Testing ReplayBuffer...")
    buffer = ReplayBuffer(capacity=1000)
    for i in range(100):
        state = np.random.randn(6)
        action = np.random.randint(0, 16)
        reward = np.random.randn()
        next_state = np.random.randn(6)
        done = np.random.random() > 0.9
        buffer.push(state, action, reward, next_state, done)
    
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"   Buffer size: {len(buffer)}")
    print(f"   Sample batch shapes:")
    print(f"     States: {states.shape}")
    print(f"     Actions: {actions.shape}")
    print(f"     Rewards: {rewards.shape}")
    print(f"   ✓ ReplayBuffer working correctly")
    
    # Test 3: DQNAgent
    print("\n3. Testing DQNAgent...")
    agent = DQNAgent(
        state_dim=6,
        action_dim=16,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=32
    )
    
    # Test observation conversion
    dummy_obs = {
        'soil_moisture': np.array([0.5]),
        'crop_stage': 1,
        'rain': np.array([0.3]),
        'et0': np.array([0.7])
    }
    state = agent.obs_to_state(dummy_obs)
    print(f"   State conversion: obs -> state of shape {state.shape}")
    print(f"   State values: {state}")
    
    # Test action selection
    action = agent.select_action(dummy_obs, evaluate=False)
    print(f"   Selected action (explore): {action}")
    
    action_greedy = agent.select_action(dummy_obs, evaluate=True)
    print(f"   Selected action (greedy): {action_greedy}")
    print(f"   ✓ Action selection working")
    
    # Test training step
    print("\n4. Testing training step...")
    # Fill buffer with dummy experiences
    for i in range(100):
        agent.replay_buffer.push(
            state=np.random.randn(6),
            action=np.random.randint(0, 16),
            reward=np.random.randn(),
            next_state=np.random.randn(6),
            done=False
        )
    
    loss = agent.train_step()
    print(f"   Training loss: {loss:.4f}")
    print(f"   Epsilon after step: {agent.epsilon:.4f}")
    print(f"   Training steps: {agent.training_steps}")
    print(f"   ✓ Training step successful")
    
    # Test save/load
    print("\n5. Testing save/load...")
    save_path = "/tmp/test_dqn_model.pt"
    agent.save(save_path)
    print(f"   Model saved to: {save_path}")
    
    agent2 = DQNAgent(state_dim=6, action_dim=16)
    agent2.load(save_path)
    print(f"   Model loaded successfully")
    print(f"   Loaded training steps: {agent2.training_steps}")
    print(f"   ✓ Save/load working correctly")
    
    print("\n" + "=" * 70)
    print("All DQN Agent Tests Passed!")
    print("=" * 70)
    print("\nAgent is ready for training with:")
    print(f"  - State dimension: {agent.state_dim}")
    print(f"  - Action dimension: {agent.action_dim}")
    print(f"  - Network architecture: 6 -> 128 -> 128 -> 16")
    print(f"  - Epsilon decay: {agent.epsilon_start} -> {agent.epsilon_end} over {agent.epsilon_decay_steps} steps")
    print(f"  - Batch size: {agent.batch_size}")
    print(f"  - Gamma: {agent.gamma}")
    print("=" * 70)
