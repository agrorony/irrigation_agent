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
