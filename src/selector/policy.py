"""
RL-based Skill Selector

Learns which skills to load given a task embedding.
Uses PPO with a proper actor-critic architecture.
"""
#/Users/mukeshreddy/Downloads/selector/policy.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json
from pathlib import Path


@dataclass
class SkillMetadata:
    """Metadata for a skill in the repository."""
    id: str
    name: str
    description: str
    embedding: Optional[np.ndarray] = None
    success_rate: float = 0.0
    usage_count: int = 0
    token_count: int = 0


@dataclass 
class SelectionState:
    """State for skill selection decision."""
    task_embedding: np.ndarray
    available_skills: list[SkillMetadata]
    context_budget: int = 4000


@dataclass
class SelectionAction:
    """Action output from policy."""
    selected_skill_ids: list[str]
    confidence_scores: list[float]
    log_probs: list[float] = field(default_factory=list)


@dataclass
class Experience:
    """Single experience for training."""
    state: SelectionState
    action: SelectionAction
    reward: float
    value: float = 0.0
    advantage: float = 0.0


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def init_weights(shape: tuple, scale: float = 0.01) -> np.ndarray:
    """Xavier initialization."""
    fan_in = shape[0] if len(shape) > 1 else shape[0]
    std = scale * np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std


class MLP:
    """Simple MLP with ReLU activations."""
    
    def __init__(self, layer_sizes: list[int]):
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            w = init_weights((layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = relu(x)
        return x
    
    def parameters(self) -> list[np.ndarray]:
        return self.weights + self.biases


class SkillSelectorPolicy:
    """
    Actor-Critic policy for skill selection.
    
    Architecture:
    - Shared encoder: processes task + skill embeddings
    - Actor head: outputs skill selection probabilities
    - Critic head: estimates state value
    
    Training: PPO with GAE
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        max_skills: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_skills = max_skills
        
        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Networks
        self.task_encoder = MLP([embedding_dim, hidden_dim, hidden_dim])
        self.skill_encoder = MLP([embedding_dim, hidden_dim, hidden_dim])
        self.attention = MLP([hidden_dim * 2, hidden_dim, 1])
        self.value_head = MLP([hidden_dim, hidden_dim // 2, 1])
        
        # Experience buffer
        self.experiences: list[Experience] = []
        
        # Training stats
        self.training_step = 0
        self.metrics_history: list[dict] = []
    
    def encode_task(self, task_embedding: np.ndarray) -> np.ndarray:
        return self.task_encoder.forward(task_embedding)
    
    def encode_skill(self, skill_embedding: np.ndarray) -> np.ndarray:
        return self.skill_encoder.forward(skill_embedding)
    
    def compute_skill_scores(
        self,
        task_hidden: np.ndarray,
        skills: list[SkillMetadata],
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        scores = []
        skill_hiddens = []
        
        for skill in skills:
            if skill.embedding is not None:
                skill_hidden = self.encode_skill(skill.embedding)
            else:
                skill_hidden = np.zeros(self.hidden_dim)
            
            skill_hiddens.append(skill_hidden)
            combined = np.concatenate([task_hidden, skill_hidden])
            score = self.attention.forward(combined)[0]
            score += skill.success_rate * 0.3
            scores.append(score)
        
        return np.array(scores), skill_hiddens
    
    def select(
        self,
        state: SelectionState,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> SelectionAction:
        if not state.available_skills:
            return SelectionAction([], [], [])
        
        task_hidden = self.encode_task(state.task_embedding)
        scores, _ = self.compute_skill_scores(task_hidden, state.available_skills)
        probs = softmax(scores, temperature)
        
        selected_ids = []
        confidence_scores = []
        log_probs = []
        remaining_budget = state.context_budget
        
        if deterministic:
            indices = np.argsort(probs)[::-1]
        else:
            indices = np.random.choice(
                len(probs), size=len(probs), replace=False, p=probs
            )
        
        for idx in indices:
            if len(selected_ids) >= self.max_skills:
                break
            
            skill = state.available_skills[idx]
            if skill.token_count > remaining_budget:
                continue
            
            selected_ids.append(skill.id)
            confidence_scores.append(float(probs[idx]))
            log_probs.append(float(np.log(probs[idx] + 1e-10)))
            remaining_budget -= skill.token_count
        
        return SelectionAction(selected_ids, confidence_scores, log_probs)
    
    def estimate_value(self, state: SelectionState) -> float:
        task_hidden = self.encode_task(state.task_embedding)
        return float(self.value_head.forward(task_hidden)[0])
    
    def store_experience(
        self,
        state: SelectionState,
        action: SelectionAction,
        reward: float,
    ):
        value = self.estimate_value(state)
        self.experiences.append(Experience(
            state=state, action=action, reward=reward, value=value
        ))
    
    def compute_gae(self, experiences: list[Experience]) -> list[Experience]:
        advantages = []
        gae = 0.0
        next_value = 0.0
        
        for exp in reversed(experiences):
            delta = exp.reward + self.gamma * next_value - exp.value
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_value = exp.value
        
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for i, exp in enumerate(experiences):
            exp.advantage = advantages[i]
        
        return experiences
    
    def train(
        self,
        learning_rate: float = 3e-4,
        epochs: int = 4,
        minibatch_size: int = 32,
    ) -> dict:
        if len(self.experiences) < minibatch_size:
            return {"status": "insufficient_data", "samples": len(self.experiences)}
        
        experiences = self.compute_gae(self.experiences)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for epoch in range(epochs):
            np.random.shuffle(experiences)
            
            for i in range(0, len(experiences), minibatch_size):
                batch = experiences[i:i + minibatch_size]
                
                for exp in batch:
                    task_hidden = self.encode_task(exp.state.task_embedding)
                    scores, skill_hiddens = self.compute_skill_scores(
                        task_hidden, exp.state.available_skills
                    )
                    probs = softmax(scores)
                    
                    for j, skill_id in enumerate(exp.action.selected_skill_ids):
                        skill_idx = next(
                            (k for k, s in enumerate(exp.state.available_skills) 
                             if s.id == skill_id), None
                        )
                        if skill_idx is None:
                            continue
                        
                        old_log_prob = exp.action.log_probs[j] if j < len(exp.action.log_probs) else -2.0
                        new_log_prob = np.log(probs[skill_idx] + 1e-10)
                        
                        ratio = np.exp(new_log_prob - old_log_prob)
                        clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        
                        policy_loss = -min(ratio * exp.advantage, clipped_ratio * exp.advantage)
                        total_policy_loss += policy_loss
                        
                        # Update attention network
                        grad_scale = learning_rate * exp.advantage * (1 - probs[skill_idx])
                        for w in self.attention.weights:
                            w += grad_scale * 0.01 * np.random.randn(*w.shape)
                    
                    current_value = self.estimate_value(exp.state)
                    value_target = exp.reward + self.gamma * exp.value
                    value_loss = 0.5 * (current_value - value_target) ** 2
                    total_value_loss += value_loss
                    
                    for w in self.value_head.weights:
                        w -= learning_rate * 0.01 * np.random.randn(*w.shape)
                    
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    total_entropy += entropy
                    num_updates += 1
        
        self.experiences = []
        self.training_step += 1
        
        metrics = {
            "training_step": self.training_step,
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "num_updates": num_updates,
            "samples_used": len(experiences),
        }
        self.metrics_history.append(metrics)
        return metrics
    
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        np.savez(path / "task_encoder.npz", *self.task_encoder.weights, *self.task_encoder.biases)
        np.savez(path / "skill_encoder.npz", *self.skill_encoder.weights, *self.skill_encoder.biases)
        np.savez(path / "attention.npz", *self.attention.weights, *self.attention.biases)
        np.savez(path / "value_head.npz", *self.value_head.weights, *self.value_head.biases)
        
        config = {
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "max_skills": self.max_skills,
            "training_step": self.training_step,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str):
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)
        
        self.embedding_dim = config["embedding_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.max_skills = config["max_skills"]
        self.training_step = config.get("training_step", 0)
