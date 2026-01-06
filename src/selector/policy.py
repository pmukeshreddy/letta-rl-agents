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

