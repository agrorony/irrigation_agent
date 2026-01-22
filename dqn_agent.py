"""
Vanilla DQN Agent for Discrete Irrigation Scheduling
=====================================================

PyTorch implementation of Deep Q-Network (DQN) with:
- Experience replay buffer
- Target network with hard updates
- Epsilon-greedy exploration with linear decay
- Huber loss for stability

No Double DQN or other enhancements for simplicity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Q-Network: MLP that processes flattened Dict observations.
    
    Input: 6-dimensional state vector
        - soil_moisture (1)
        - crop_stage_onehot (3)
        - rain (1)
        - et0 (1)
    
    Output: Q-values for each discrete action
    
    Parameters
    ----------
    state_dim : int
        Dimension of flattened state (default: 6)
    n_actions : int
        Number of discrete actions (default: 16)
    hidden_dims : list of int
        Hidden layer dimensions (default: [128, 128])
    """
    
    def __init__(self, state_dim=6, n_actions=16, hidden_dims=[128, 128]):
        super(QNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        
        # Build MLP layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer for Q-values
        layers.append(nn.Linear(prev_dim, n_actions))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Forward pass through Q-network.
        
        Parameters
        ----------
        state : torch.Tensor
            Flattened state tensor of shape (batch_size, state_dim)
        
        Returns
        -------
        q_values : torch.Tensor
            Q-values for each action, shape (batch_size, n_actions)
        """
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer with uniform sampling.
    
    Stores transitions (state, action, reward, next_state, done) and
    provides random sampling for training.
    
    Parameters
    ----------
    capacity : int
        Maximum number of transitions to store
    """
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Parameters
        ----------
        state : np.ndarray
            Current state (flattened)
        action : int
            Action taken
        reward : float
            Reward received
        next_state : np.ndarray
            Next state (flattened)
        done : bool
            Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Parameters
        ----------
        batch_size : int
            Number of transitions to sample
        
        Returns
        -------
        batch : tuple of np.ndarray
            (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Vanilla DQN agent with epsilon-greedy exploration and target network.
    
    Parameters
    ----------
    state_dim : int
        Dimension of flattened state
    n_actions : int
        Number of discrete actions
    learning_rate : float
        Learning rate for optimizer (default: 1e-4)
    gamma : float
        Discount factor (default: 0.99)
    epsilon_start : float
        Initial exploration rate (default: 1.0)
    epsilon_end : float
        Final exploration rate (default: 0.05)
    epsilon_decay_steps : int
        Number of steps to decay epsilon (default: 50000)
    buffer_size : int
        Replay buffer capacity (default: 100000)
    batch_size : int
        Training batch size (default: 64)
    target_update_freq : int
        Frequency of target network updates in steps (default: 1000)
    hidden_dims : list of int
        Hidden layer dimensions (default: [128, 128])
    device : str
        Device to use ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        state_dim=6,
        n_actions=16,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        hidden_dims=[128, 128],
        device='cpu'
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        
        # Epsilon-greedy parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = epsilon_start
        self.steps_done = 0
        
        # Networks
        self.q_network = QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Training stats
        self.total_updates = 0
    
    def obs_to_state(self, obs):
        """
        Convert Dict observation to flattened state vector.
        
        The environment's observation space has crop_stage as a Discrete(3) space,
        meaning it's returned as an integer (0, 1, or 2) rather than an array.
        This is consistent with Gymnasium's design where Discrete spaces return scalars.
        We one-hot encode it here for the neural network input.
        
        Parameters
        ----------
        obs : dict
            Observation from environment with keys:
            - 'soil_moisture': array([value])
            - 'crop_stage': int (0, 1, or 2) - Discrete space returns scalar
            - 'rain': array([value])
            - 'et0': array([value])
        
        Returns
        -------
        state : np.ndarray
            Flattened state vector of shape (6,)
        """
        # One-hot encode crop_stage
        crop_stage_onehot = np.zeros(3, dtype=np.float32)
        crop_stage_onehot[obs['crop_stage']] = 1.0
        
        state = np.concatenate([
            obs['soil_moisture'],      # 1 dim
            crop_stage_onehot,         # 3 dims (one-hot)
            obs['rain'],               # 1 dim
            obs['et0']                 # 1 dim
        ], axis=0)
        return state.astype(np.float32)
    
    def select_action(self, obs, epsilon=None):
        """
        Select action using epsilon-greedy policy.
        
        Parameters
        ----------
        obs : dict
            Observation from environment
        epsilon : float, optional
            Exploration rate (if None, uses agent's current epsilon)
        
        Returns
        -------
        action : int
            Selected action index
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        else:
            state = self.obs_to_state(obs)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            return q_values.argmax(dim=1).item()
    
    def update_epsilon(self):
        """Update epsilon using linear decay."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.epsilon_start - self.epsilon_end) * 
            (self.steps_done / self.epsilon_decay_steps)
        )
        self.steps_done += 1
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store transition in replay buffer.
        
        Parameters
        ----------
        obs : dict
            Current observation
        action : int
            Action taken
        reward : float
            Reward received
        next_obs : dict
            Next observation
        done : bool
            Whether episode terminated
        """
        state = self.obs_to_state(obs)
        next_state = self.obs_to_state(next_obs)
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step using a batch from replay buffer.
        
        Returns
        -------
        loss : float or None
            Training loss (None if buffer is too small)
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
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values (vanilla DQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.total_updates += 1
        
        # Update target network
        if self.total_updates % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def save(self, path):
        """
        Save agent state to file.
        
        Parameters
        ----------
        path : str
            Path to save checkpoint
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'total_updates': self.total_updates,
            'state_dim': self.state_dim,
            'n_actions': self.n_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq
        }, path)
    
    def load(self, path):
        """
        Load agent state from file.
        
        Parameters
        ----------
        path : str
            Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.total_updates = checkpoint['total_updates']


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("DQN Agent - Demo")
    print("=" * 70)
    
    # Create agent
    agent = DQNAgent(
        state_dim=6,
        n_actions=16,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50000,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        hidden_dims=[128, 128]
    )
    
    print(f"\nAgent Configuration:")
    print(f"  State dim: {agent.state_dim}")
    print(f"  N actions: {agent.n_actions}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Epsilon: {agent.epsilon:.3f} -> {agent.epsilon_end:.3f}")
    print(f"  Buffer size: {len(agent.replay_buffer)}")
    print(f"  Batch size: {agent.batch_size}")
    print(f"  Target update freq: {agent.target_update_freq}")
    
    print(f"\nQ-Network Architecture:")
    print(agent.q_network)
    
    # Test observation processing
    obs = {
        'soil_moisture': np.array([0.5]),
        'crop_stage': 1,  # Integer, not array
        'rain': np.array([0.3]),
        'et0': np.array([0.7])
    }
    
    state = agent.obs_to_state(obs)
    print(f"\nFlattened state: {state}")
    print(f"State shape: {state.shape}")
    
    # Test action selection
    action = agent.select_action(obs)
    print(f"\nSelected action (epsilon={agent.epsilon:.3f}): {action}")
    
    # Test greedy action
    action_greedy = agent.select_action(obs, epsilon=0.0)
    print(f"Greedy action (epsilon=0.0): {action_greedy}")
    
    print("\n" + "=" * 70)
    print("DQN agent ready for training!")
    print("=" * 70)
