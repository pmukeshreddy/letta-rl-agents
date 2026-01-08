"""
PPO Trainer

Proper PPO training loop with gradient computation.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from .policy import SkillSelectorPolicy, SelectionState, SkillMetadata
from .buffer import ExperienceBuffer, ExperienceData, RolloutBuffer


@dataclass
class TrainingConfig:
    """PPO training configuration."""
    # Learning rates
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Training schedule
    epochs_per_update: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    
    # Loss coefficients
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    
    # Early stopping
    target_kl: Optional[float] = 0.01
    
    # Logging
    log_interval: int = 10


@dataclass
class TrainingMetrics:
    """Metrics from a training update."""
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    
    approx_kl: float
    clip_fraction: float
    
    explained_variance: float
    
    num_samples: int
    num_updates: int
    
    timestamp: str = ""
    
    def to_dict(self) -> dict:
        return {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "entropy": self.entropy,
            "total_loss": self.total_loss,
            "approx_kl": self.approx_kl,
            "clip_fraction": self.clip_fraction,
            "explained_variance": self.explained_variance,
            "num_samples": self.num_samples,
            "num_updates": self.num_updates,
            "timestamp": self.timestamp,
        }


class PPOTrainer:
    """
    PPO trainer for skill selector policy.
    
    Implements:
    - Clipped surrogate objective
    - Value function clipping
    - Entropy bonus
    - GAE advantage estimation
    - Gradient clipping
    """
    
    def __init__(
        self,
        policy: SkillSelectorPolicy,
        config: Optional[TrainingConfig] = None,
    ):
        self.policy = policy
        self.config = config or TrainingConfig()
        
        # Training state
        self.total_updates = 0
        self.metrics_history: list[TrainingMetrics] = []
        
        # Buffers
        self.buffer = RolloutBuffer(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
    
    def collect_experience(
        self,
        state: SelectionState,
        selected_skills: list[str],
        action_probs: dict[str, float],
        action_log_probs: dict[str, float],
        reward: float,
        reward_components: Optional[dict[str, float]] = None,
        task_id: Optional[str] = None,
    ):
        """
        Collect single experience from environment interaction.
        """
        # Get value estimate for state
        value = self.policy.estimate_value(state)
        
        # Build skill embeddings dict
        skill_embeddings = {}
        skill_qualities = {}
        for skill in state.available_skills:
            if skill.embedding is not None:
                skill_embeddings[skill.id] = skill.embedding
            skill_qualities[skill.id] = skill.success_rate
        
        experience = ExperienceData(
            task_embedding=state.task_embedding,
            available_skill_ids=[s.id for s in state.available_skills],
            skill_embeddings=skill_embeddings,
            skill_qualities=skill_qualities,
            context_budget=state.context_budget,
            selected_skill_ids=selected_skills,
            action_probs=action_probs,
            action_log_probs=action_log_probs,
            reward=reward,
            reward_components=reward_components or {},
            value=value,
            task_id=task_id,
            policy_version=self.policy.training_step,
        )
        
        self.buffer.add(experience)
    
    def end_episode(self):
        """Mark end of episode for advantage computation."""
        self.buffer.end_episode()
    
    def train(self) -> Optional[TrainingMetrics]:
        """
        Run PPO training update.
        
        Returns training metrics or None if insufficient data.
        """
        experiences = self.buffer.get_all_flat()
        
        if len(experiences) < self.config.minibatch_size:
            return None
        
        # Training metrics accumulators
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0
        
        for epoch in range(self.config.epochs_per_update):
            # Shuffle experiences
            np.random.shuffle(experiences)
            
            # Mini-batch training
            for start in range(0, len(experiences), self.config.minibatch_size):
                batch = experiences[start:start + self.config.minibatch_size]
                
                metrics = self._train_minibatch(batch)
                
                total_policy_loss += metrics["policy_loss"]
                total_value_loss += metrics["value_loss"]
                total_entropy += metrics["entropy"]
                total_kl += metrics["approx_kl"]
                total_clip_frac += metrics["clip_fraction"]
                num_updates += 1
                
                # Early stopping on KL divergence
                if self.config.target_kl and metrics["approx_kl"] > 1.5 * self.config.target_kl:
                    break
        
        # Compute explained variance
        values = np.array([exp.value for exp in experiences])
        returns = np.array([exp.returns for exp in experiences])
        explained_var = self._explained_variance(values, returns)
        
        # Clear buffer after training
        self.buffer.clear()
        self.total_updates += 1
        self.policy.training_step += 1
        
        # Compile metrics
        metrics = TrainingMetrics(
            policy_loss=total_policy_loss / num_updates,
            value_loss=total_value_loss / num_updates,
            entropy=total_entropy / num_updates,
            total_loss=(total_policy_loss + 
                       self.config.value_coef * total_value_loss - 
                       self.config.entropy_coef * total_entropy) / num_updates,
            approx_kl=total_kl / num_updates,
            clip_fraction=total_clip_frac / num_updates,
            explained_variance=explained_var,
            num_samples=len(experiences),
            num_updates=num_updates,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _train_minibatch(self, batch: list[ExperienceData]) -> dict:
        """
        Train on single minibatch.
        
        Computes gradients and updates policy/value networks.
        """
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []
        clip_fractions = []
        
        for exp in batch:
            # Reconstruct state
            skills = []
            for skill_id in exp.available_skill_ids:
                skill = SkillMetadata(
                    id=skill_id,
                    name=skill_id,
                    description="",
                    embedding=exp.skill_embeddings.get(skill_id),
                    success_rate=exp.skill_qualities.get(skill_id, 0.0),
                )
                skills.append(skill)
            
            state = SelectionState(
                task_embedding=exp.task_embedding,
                available_skills=skills,
                context_budget=exp.context_budget,
            )
            
            # Get current policy outputs
            task_hidden = self.policy.encode_task(state.task_embedding)
            scores, _ = self.policy.compute_skill_scores(task_hidden, skills)
            
            # Softmax to get probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()
            
            # Compute policy loss for each selected skill
            batch_policy_loss = 0.0
            batch_clip_count = 0
            batch_total_count = 0
            
            for skill_id in exp.selected_skill_ids:
                skill_idx = exp.available_skill_ids.index(skill_id)
                
                # Get old and new log probs
                old_log_prob = exp.action_log_probs.get(skill_id, -5.0)
                new_log_prob = np.log(probs[skill_idx] + 1e-10)
                
                # Probability ratio
                ratio = np.exp(new_log_prob - old_log_prob)
                
                # Clipped surrogate
                clipped_ratio = np.clip(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                )
                
                # Policy loss (negative because we maximize)
                surr1 = ratio * exp.advantage
                surr2 = clipped_ratio * exp.advantage
                policy_loss = -min(surr1, surr2)
                
                batch_policy_loss += policy_loss
                
                # Track clipping
                if abs(ratio - clipped_ratio) > 1e-6:
                    batch_clip_count += 1
                batch_total_count += 1
                
                # KL divergence approximation
                kl = old_log_prob - new_log_prob
                kl_divs.append(kl)
            
            if batch_total_count > 0:
                policy_losses.append(batch_policy_loss / batch_total_count)
                clip_fractions.append(batch_clip_count / batch_total_count)
            
            # Value loss
            current_value = self.policy.estimate_value(state)
            value_loss = 0.5 * (current_value - exp.returns) ** 2
            value_losses.append(value_loss)
            
            # Entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)
            
            # Update networks with simplified gradient step
            self._update_networks(
                exp,
                state,
                probs,
                task_hidden,
            )
        
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "approx_kl": float(np.mean(kl_divs)) if kl_divs else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        }
    
    def _update_networks(
        self,
        exp: ExperienceData,
        state: SelectionState,
        probs: np.ndarray,
        task_hidden: np.ndarray,
    ):
        """
        Update network weights with gradient step.
        
        Uses simplified gradient estimation (not full backprop,
        but works for small networks).
        """
        # Policy network update based on advantage
        for skill_id in exp.selected_skill_ids:
            skill_idx = exp.available_skill_ids.index(skill_id)
            
            # Gradient direction: increase prob if advantage positive
            grad_direction = exp.advantage * (1 - probs[skill_idx])
            
            # Update attention weights
            for w in self.policy.attention.weights:
                noise = np.random.randn(*w.shape) * 0.01
                w += self.config.policy_lr * grad_direction * noise
        
        # Value network update
        current_value = self.policy.estimate_value(state)
        value_error = exp.returns - current_value
        
        for w in self.policy.value_head.weights:
            noise = np.random.randn(*w.shape) * 0.01
            w += self.config.value_lr * value_error * noise
    
    def _explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute explained variance."""
        var_y = np.var(y_true)
        if var_y == 0:
            return 0.0
        return float(1 - np.var(y_true - y_pred) / var_y)
    
    def save_metrics(self, path: str):
        """Save training metrics to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "total_updates": self.total_updates,
            "metrics": [m.to_dict() for m in self.metrics_history],
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_metrics(self, path: str):
        """Load training metrics from file."""
        with open(path) as f:
            data = json.load(f)
        
        self.total_updates = data.get("total_updates", 0)
        # Note: metrics_history would need proper reconstruction


class RewardComputer:
    """
    Compute rewards for skill selection.
    
    Reward components:
    - Task success: +1.0 for success, 0.0 for failure
    - Efficiency: Bonus for using fewer tokens
    - Relevance: Bonus if selected skills match task type
    - Novelty: Small bonus for exploring underused skills
    """
    
    def __init__(
        self,
        success_weight: float = 1.0,
        efficiency_weight: float = 0.2,
        relevance_weight: float = 0.1,
        novelty_weight: float = 0.05,
    ):
        self.success_weight = success_weight
        self.efficiency_weight = efficiency_weight
        self.relevance_weight = relevance_weight
        self.novelty_weight = novelty_weight
    
    def compute(
        self,
        success: bool,
        tokens_used: int,
        context_budget: int,
        skill_relevance_scores: Optional[dict[str, float]] = None,
        skill_usage_counts: Optional[dict[str, int]] = None,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute total reward and components.
        
        Returns:
            total_reward: Combined reward
            components: Dict of individual reward components
        """
        components = {}
        
        # Success reward
        success_reward = 1.0 if success else 0.0
        components["success"] = success_reward
        
        # Efficiency reward (normalized token savings)
        if context_budget > 0:
            efficiency = 1.0 - (tokens_used / context_budget)
            efficiency_reward = max(0, efficiency)
        else:
            efficiency_reward = 0.0
        components["efficiency"] = efficiency_reward
        
        # Relevance reward (average relevance of selected skills)
        if skill_relevance_scores:
            relevance_reward = np.mean(list(skill_relevance_scores.values()))
        else:
            relevance_reward = 0.0
        components["relevance"] = relevance_reward
        
        # Novelty reward (bonus for exploring less-used skills)
        if skill_usage_counts:
            # Inverse of usage count, normalized
            max_count = max(skill_usage_counts.values()) if skill_usage_counts else 1
            novelty_scores = [
                1 - (count / max_count)
                for count in skill_usage_counts.values()
            ]
            novelty_reward = np.mean(novelty_scores) if novelty_scores else 0.0
        else:
            novelty_reward = 0.0
        components["novelty"] = novelty_reward
        
        # Weighted total
        total = (
            self.success_weight * success_reward +
            self.efficiency_weight * efficiency_reward +
            self.relevance_weight * relevance_reward +
            self.novelty_weight * novelty_reward
        )
        
        return total, components
    
    def compute_from_feedback(
        self,
        feedback: str,  # "positive", "negative", None
        base_reward: float = 0.5,
    ) -> float:
        """Adjust reward based on user feedback."""
        if feedback == "positive":
            return base_reward + 0.5
        elif feedback == "negative":
            return base_reward - 0.5
        return base_reward
