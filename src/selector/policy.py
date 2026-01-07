from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import json
from pathlib import Path


@dataclass
class SkillMetaData:
    id:str
    name : str
    description : str
    embedding : Optional[np.ndarray] = None
    success_rate : float = 0.0
    usage_count : int = 0
    token_count : int = 0

@dataclass 
class SelectionState:
    task_embedding: np.ndarray
    available_skills: list[SkillMetadata]
    context_budget: int = 4000


@dataclass
class SelectionAction:
    selected_skill_ids = list[str]
    confidence_scores = list[float]
    log_probs = list[float] = field(default_factory=list)


@dataclass
class Experience:
    state: SelectionState
    action: SelectionAction
    reward: float
    value: float = 0.0
    advantage: float = 0.0

def relu(x:float)->float:
    retun max(0,x)

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def init_weights(shape: tuple, scale: float = 0.01) -> np.ndarray:
    fan_in = shape[1] if len(shape) > 1 else shape[0]
    std = scale * np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std    

class MLP:
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