"""
Experience Buffer

Stores experiences for PPO training with proper advantage estimation.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from collections import deque


@dataclass
class ExperienceData:
    """
    Single experience for training.
    
    Contains all information needed for PPO update.
    """
    # State
    task_embedding: np.ndarray
    available_skill_ids: list[str]
    skill_embeddings: dict[str, np.ndarray]  # skill_id -> embedding
    skill_qualities: dict[str, float]  # skill_id -> success_rate
    context_budget: int
    
    # Action
    selected_skill_ids: list[str]
    action_probs: dict[str, float]  # skill_id -> selection probability
    action_log_probs: dict[str, float]  # skill_id -> log probability
    
    # Outcome
    reward: float
    reward_components: dict[str, float] = field(default_factory=dict)
    
    # Value estimation (filled during training)
    value: float = 0.0
    advantage: float = 0.0
    returns: float = 0.0
    
    # Metadata
    task_id: Optional[str] = None
    policy_version: int = 0
    timestamp: Optional[float] = None


class ExperienceBuffer:
    """
    Fixed-size buffer for storing experiences.
    
    Features:
    - FIFO when full
    - Batch sampling for training
    - GAE advantage computation
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.max_size = max_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.buffer: deque[ExperienceData] = deque(maxlen=max_size)
        self._needs_advantage_update = True
    
    def add(self, experience: ExperienceData):
        """Add experience to buffer."""
        self.buffer.append(experience)
        self._needs_advantage_update = True
    
    def add_batch(self, experiences: list[ExperienceData]):
        """Add multiple experiences."""
        for exp in experiences:
            self.add(exp)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self._needs_advantage_update = True
    
    def get_all(self) -> list[ExperienceData]:
        """Get all experiences (for full batch training)."""
        if self._needs_advantage_update:
            self._compute_advantages()
        return list(self.buffer)
    
    def sample(self, batch_size: int) -> list[ExperienceData]:
        """Sample random batch of experiences."""
        if self._needs_advantage_update:
            self._compute_advantages()
        
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_recent(self, n: int) -> list[ExperienceData]:
        """Sample most recent n experiences."""
        if self._needs_advantage_update:
            self._compute_advantages()
        
        return list(self.buffer)[-n:]
    
    def _compute_advantages(self):
        """
        Compute GAE advantages for all experiences.
        
        GAE(t) = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        if len(self.buffer) == 0:
            return
        
        experiences = list(self.buffer)
        
        # Compute returns and advantages in reverse order
        gae = 0.0
        next_value = 0.0
        
        for i in reversed(range(len(experiences))):
            exp = experiences[i]
            
            # TD error: delta = r + gamma * V(next) - V(current)
            delta = exp.reward + self.gamma * next_value - exp.value
            
            # GAE: advantage = delta + gamma * lambda * prev_advantage
            gae = delta + self.gamma * self.gae_lambda * gae
            
            # Store
            exp.advantage = gae
            exp.returns = gae + exp.value
            
            next_value = exp.value
        
        # Normalize advantages
        advantages = np.array([exp.advantage for exp in experiences])
        mean = advantages.mean()
        std = advantages.std() + 1e-8
        
        for exp in experiences:
            exp.advantage = (exp.advantage - mean) / std
        
        self._needs_advantage_update = False
    
    def get_stats(self) -> dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "avg_reward": 0.0,
                "avg_advantage": 0.0,
                "reward_std": 0.0,
            }
        
        rewards = [exp.reward for exp in self.buffer]
        advantages = [exp.advantage for exp in self.buffer]
        
        return {
            "size": len(self.buffer),
            "avg_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "avg_advantage": float(np.mean(advantages)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }


class PrioritizedBuffer(ExperienceBuffer):
    """
    Prioritized experience buffer.
    
    Samples experiences based on TD error magnitude.
    Higher error = more learning potential = higher priority.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,  # Importance sampling
        beta_increment: float = 0.001,
    ):
        super().__init__(max_size, gamma, gae_lambda)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.priorities: deque[float] = deque(maxlen=max_size)
    
    def add(self, experience: ExperienceData, priority: Optional[float] = None):
        """Add experience with optional priority."""
        super().add(experience)
        
        # Default to max priority for new experiences
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> tuple[list[ExperienceData], np.ndarray, list[int]]:
        """
        Sample batch with prioritization.
        
        Returns:
            experiences: Sampled experiences
            weights: Importance sampling weights
            indices: Indices for priority updates
        """
        if self._needs_advantage_update:
            self._compute_advantages()
        
        n = len(self.buffer)
        if n <= batch_size:
            return list(self.buffer), np.ones(n), list(range(n))
        
        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(n, batch_size, replace=False, p=probs)
        
        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increment beta towards 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        experiences = [self.buffer[i] for i in indices]
        return experiences, weights, indices.tolist()
    
    def update_priorities(self, indices: list[int], td_errors: list[float]):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
    
    def clear(self):
        """Clear buffer and priorities."""
        super().clear()
        self.priorities.clear()


class RolloutBuffer:
    """
    Episode-based buffer for on-policy algorithms.
    
    Stores complete episodes and clears after each training update.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.episodes: list[list[ExperienceData]] = []
        self.current_episode: list[ExperienceData] = []
    
    def add(self, experience: ExperienceData):
        """Add experience to current episode."""
        self.current_episode.append(experience)
    
    def end_episode(self):
        """End current episode and compute returns."""
        if self.current_episode:
            self._compute_episode_advantages(self.current_episode)
            self.episodes.append(self.current_episode)
            self.current_episode = []
    
    def _compute_episode_advantages(self, episode: list[ExperienceData]):
        """Compute GAE for single episode."""
        gae = 0.0
        next_value = 0.0
        
        for exp in reversed(episode):
            delta = exp.reward + self.gamma * next_value - exp.value
            gae = delta + self.gamma * self.gae_lambda * gae
            exp.advantage = gae
            exp.returns = gae + exp.value
            next_value = exp.value
    
    def get_all_flat(self) -> list[ExperienceData]:
        """Get all experiences flattened."""
        # End any ongoing episode
        if self.current_episode:
            self.end_episode()
        
        # Flatten episodes
        all_experiences = []
        for episode in self.episodes:
            all_experiences.extend(episode)
        
        # Normalize advantages across all data
        if all_experiences:
            advantages = np.array([exp.advantage for exp in all_experiences])
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            for exp in all_experiences:
                exp.advantage = (exp.advantage - mean) / std
        
        return all_experiences
    
    def clear(self):
        """Clear all data."""
        self.episodes = []
        self.current_episode = []
    
    def __len__(self) -> int:
        total = sum(len(ep) for ep in self.episodes)
        total += len(self.current_episode)
        return total
    
    @property
    def num_episodes(self) -> int:
        return len(self.episodes) + (1 if self.current_episode else 0)
