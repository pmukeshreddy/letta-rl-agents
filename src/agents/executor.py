"""
Task Executor

Orchestrates the full pipeline:
Embeddings → Policy → Skill Loading → Letta Execution → Reward → Training
"""

from dataclasses import dataclass
from typing import Optional
import uuid
from datetime import datetime

from ..selector.embeddings import EmbeddingModel, SkillEmbedder
from ..selector.policy import SkillSelectorPolicy, SelectionState, SelectionAction, SkillMetadata
from ..selector.trainer import PPOTrainer, TrainingConfig, RewardComputer
from ..selector.buffer import ExperienceData


@dataclass
class Skill:
    """Skill with full content."""
    id: str
    name: str
    description: str
    content: str
    category: Optional[str] = None
    token_count: int = 0
    success_rate: float = 0.0
    usage_count: int = 0


@dataclass
class TaskResult:
    """Result from task execution."""
    task_id: str
    success: bool
    response: str
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: int = 0
    skills_used: list[str] = None
    reward: float = 0.0
    reward_components: dict = None


class SkillSelector:
    """
    Complete skill selection system.
    
    Connects:
    - Embeddings (task/skill vectors)
    - Policy (RL selection)
    - Trainer (PPO updates)
    - Reward computation
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        max_skills: int = 5,
        context_budget: int = 4000,
        training_config: Optional[TrainingConfig] = None,
    ):
        # Embeddings
        self.embedding_model = EmbeddingModel(model_name=embedding_model)
        self.embedder = SkillEmbedder(self.embedding_model)
        
        # Policy
        self.policy = SkillSelectorPolicy(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            max_skills=max_skills,
        )
        
        # Trainer
        self.trainer = PPOTrainer(
            policy=self.policy,
            config=training_config or TrainingConfig(),
        )
        
        # Reward computer
        self.reward_computer = RewardComputer()
        
        # Config
        self.context_budget = context_budget
        
        # Skill cache (id -> embedding)
        self._skill_embeddings: dict[str, any] = {}
        self._skill_metadata: dict[str, SkillMetadata] = {}
    
    def register_skill(self, skill: Skill):
        """
        Register a skill and compute its embedding.
        
        Call this once per skill when loading from repository.
        """
        # Compute embedding
        embedding = self.embedder.embed_skill(skill.name, skill.content)
        self._skill_embeddings[skill.id] = embedding
        
        # Store metadata
        self._skill_metadata[skill.id] = SkillMetadata(
            id=skill.id,
            name=skill.name,
            description=skill.description,
            embedding=embedding,
            success_rate=skill.success_rate,
            usage_count=skill.usage_count,
            token_count=skill.token_count,
        )
    
    def register_skills(self, skills: list[Skill]):
        """Register multiple skills."""
        for skill in skills:
            self.register_skill(skill)
    
    def update_skill_stats(self, skill_id: str, success_rate: float, usage_count: int):
        """Update skill statistics (call after DB update)."""
        if skill_id in self._skill_metadata:
            self._skill_metadata[skill_id].success_rate = success_rate
            self._skill_metadata[skill_id].usage_count = usage_count
    
    def select(
        self,
        task_description: str,
        available_skill_ids: Optional[list[str]] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> tuple[list[str], dict[str, float], dict[str, float]]:
        """
        Select skills for a task.
        
        Args:
            task_description: What the task is
            available_skill_ids: Subset of skills to consider (None = all)
            temperature: Exploration temperature
            deterministic: If True, always pick top skills
            
        Returns:
            selected_ids: List of selected skill IDs
            probabilities: Dict of skill_id -> selection probability
            log_probs: Dict of skill_id -> log probability
        """
        # Embed task
        task_embedding = self.embedder.embed_task(task_description)
        
        # Get available skills
        if available_skill_ids:
            skills = [self._skill_metadata[sid] for sid in available_skill_ids if sid in self._skill_metadata]
        else:
            skills = list(self._skill_metadata.values())
        
        if not skills:
            return [], {}, {}
        
        # Build state
        state = SelectionState(
            task_embedding=task_embedding,
            available_skills=skills,
            context_budget=self.context_budget,
        )
        
        # Policy selects
        action = self.policy.select(state, temperature, deterministic)
        
        # Build probability dicts
        probs = {sid: conf for sid, conf in zip(action.selected_skill_ids, action.confidence_scores)}
        log_probs = {sid: lp for sid, lp in zip(action.selected_skill_ids, action.log_probs)}
        
        # Store state for later training
        self._last_state = state
        self._last_action = action
        self._last_probs = probs
        self._last_log_probs = log_probs
        
        return action.selected_skill_ids, probs, log_probs
    
    def record_outcome(
        self,
        success: bool,
        tokens_used: int = 0,
        task_id: Optional[str] = None,
        user_feedback: Optional[str] = None,
    ) -> float:
        """
        Record task outcome and compute reward.
        
        Call this after task execution to enable learning.
        
        Returns:
            reward: Computed reward value
        """
        if not hasattr(self, '_last_state'):
            return 0.0
        
        # Compute reward
        reward, components = self.reward_computer.compute(
            success=success,
            tokens_used=tokens_used,
            context_budget=self.context_budget,
        )
        
        # Adjust for user feedback
        if user_feedback:
            feedback_adj = self.reward_computer.compute_from_feedback(user_feedback, reward)
            reward = feedback_adj
        
        # Collect experience for training
        self.trainer.collect_experience(
            state=self._last_state,
            selected_skills=self._last_action.selected_skill_ids,
            action_probs=self._last_probs,
            action_log_probs=self._last_log_probs,
            reward=reward,
            reward_components=components,
            task_id=task_id,
        )
        
        return reward
    
    def end_episode(self):
        """Mark end of episode (for advantage computation)."""
        self.trainer.end_episode()
    
    def train(self) -> Optional[dict]:
        """
        Run training update if enough experiences.
        
        Returns training metrics or None.
        """
        metrics = self.trainer.train()
        if metrics:
            return metrics.to_dict()
        return None
    
    def save(self, path: str):
        """Save policy weights."""
        self.policy.save(path)
    
    def load(self, path: str):
        """Load policy weights."""
        self.policy.load(path)


class TaskExecutor:
    """
    Full task execution pipeline.
    
    Connects:
    - SkillSelector (our RL layer)
    - LettaClient (agent execution)
    - SkillRepository (skill storage)
    """
    
    def __init__(
        self,
        selector: SkillSelector,
        letta_client: "LettaClient",  # Forward reference
        skill_repository: "SkillRepository",  # Forward reference
    ):
        self.selector = selector
        self.letta = letta_client
        self.skills = skill_repository
        
        # Load skills into selector
        self._load_skills()
    
    def _load_skills(self):
        """Load all skills from repository into selector."""
        all_skills = self.skills.list_all()
        
        for skill_record in all_skills:
            skill = Skill(
                id=skill_record.id,
                name=skill_record.name,
                description=skill_record.description or "",
                content=skill_record.content,
                category=skill_record.category,
                token_count=skill_record.token_count,
                success_rate=skill_record.success_rate,
                usage_count=skill_record.total_uses,
            )
            self.selector.register_skill(skill)
    
    def execute(
        self,
        task_description: str,
        agent_id: Optional[str] = None,
        temperature: float = 1.0,
    ) -> TaskResult:
        """
        Execute a task with RL-selected skills.
        
        Full pipeline:
        1. Select skills (RL policy)
        2. Load skill content
        3. Execute via Letta
        4. Record outcome
        5. Update stats
        """
        task_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        # 1. Select skills
        selected_ids, probs, log_probs = self.selector.select(
            task_description=task_description,
            temperature=temperature,
        )
        
        # 2. Get skill content
        skill_contents = []
        for skill_id in selected_ids:
            skill = self.skills.get(skill_id)
            if skill:
                skill_contents.append({
                    "id": skill.id,
                    "name": skill.name,
                    "content": skill.content,
                })
        
        # 3. Execute via Letta
        try:
            response = self.letta.execute_with_skills(
                task=task_description,
                skills=skill_contents,
                agent_id=agent_id,
            )
            success = response.get("success", False)
            response_text = response.get("response", "")
            error = response.get("error")
            tokens_used = response.get("tokens_used", 0)
            
        except Exception as e:
            success = False
            response_text = ""
            error = str(e)
            tokens_used = 0
        
        # 4. Record outcome
        latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        reward = self.selector.record_outcome(
            success=success,
            tokens_used=tokens_used,
            task_id=task_id,
        )
        
        # 5. Update skill stats in repository
        for skill_id in selected_ids:
            self.skills.update_stats(
                skill_id=skill_id,
                success=success,
                reward=reward,
            )
            # Also update selector's cache
            skill = self.skills.get(skill_id)
            if skill:
                self.selector.update_skill_stats(
                    skill_id=skill_id,
                    success_rate=skill.success_rate,
                    usage_count=skill.total_uses,
                )
        
        return TaskResult(
            task_id=task_id,
            success=success,
            response=response_text,
            error=error,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            skills_used=selected_ids,
            reward=reward,
            reward_components={"success": 1.0 if success else 0.0},
        )
    
    def train(self) -> Optional[dict]:
        """Run training update."""
        self.selector.end_episode()
        return self.selector.train()
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "policy_version": self.selector.policy.training_step,
            "skills_registered": len(self.selector._skill_metadata),
            "buffer_size": len(self.selector.trainer.buffer),
        }
