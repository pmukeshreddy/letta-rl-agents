"""
Letta Agent Wrapper

Wraps Letta client with RL skill selection.
"""

import os
import uuid
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pathlib import Path
import numpy as np

from src.selector.policy import SkillSelectorPolicy, SelectionState, SkillMetadata
from src.selector.embeddings import EmbeddingModel, SkillEmbedder
from src.tracker.quality import SkillTracker, TaskOutcome


@dataclass
class Skill:
    """A skill file."""
    id: str
    name: str
    content: str
    path: str
    token_count: int = 0
    embedding: Optional[np.ndarray] = None


@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    success: bool
    response: str
    skills_used: list[str]
    confidence_scores: list[float]
    tokens_used: int
    latency_ms: int = 0
    error: Optional[str] = None


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    model: str = "openai/gpt-4o-mini"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_skills: int = 3
    context_budget: int = 4000
    use_rl_selection: bool = True
    temperature: float = 1.0


class LettaRLAgent:
    """
    Letta agent enhanced with RL skill selection.
    
    Flow:
    1. Task comes in
    2. Embed task using local model
    3. RL policy selects optimal skills
    4. Skills loaded into agent context
    5. Agent executes task via Letta API
    6. Outcome feeds back to improve policy
    """
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        letta_api_key: Optional[str] = None,
        letta_base_url: str = "https://api.letta.com",
        skills_dir: Optional[str] = None,
        policy_path: Optional[str] = None,
    ):
        self.config = config or AgentConfig()
        self.api_key = letta_api_key or os.getenv("LETTA_API_KEY")
        self.base_url = letta_base_url
        
        # RL components
        self.policy = SkillSelectorPolicy(
            embedding_dim=384,  # MiniLM dimension
            max_skills=self.config.max_skills,
        )
        self.tracker = SkillTracker()
        
        # Embeddings
        self.embedder = SkillEmbedder(EmbeddingModel(self.config.embedding_model))
        
        # Skill repository
        self.skills: dict[str, Skill] = {}
        
        # Letta client (lazy init)
        self._client = None
        self._agent_id = None
        self._agent_created = False
        
        # Load skills if directory provided
        if skills_dir:
            self.load_skills_from_dir(skills_dir)
        
        # Load policy if path provided
        if policy_path and Path(policy_path).exists():
            self.policy.load(policy_path)
    
    @property
    def client(self):
        """Lazy init Letta client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("LETTA_API_KEY not set")
            try:
                from letta_client import Letta
                self._client = Letta(api_key=self.api_key)
            except ImportError:
                raise ImportError("Install letta-client: pip install letta-client")
        return self._client
    
    def register_skill(self, skill: Skill):
        """Register a skill and compute its embedding."""
        # Compute embedding
        skill.embedding = self.embedder.embed_skill(skill.name, skill.content)
        
        # Estimate token count (rough: 4 chars per token)
        skill.token_count = len(skill.content) // 4
        
        self.skills[skill.id] = skill
    
    def load_skills_from_dir(self, dir_path: str):
        """Load all .md skills from a directory."""
        import glob
        
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return
        
        for path in glob.glob(str(dir_path / "*.md")):
            skill_id = Path(path).stem
            with open(path) as f:
                content = f.read()
            
            # Extract name from first header
            lines = content.split("\n")
            name = skill_id
            for line in lines:
                if line.startswith("# "):
                    name = line[2:].strip()
                    break
            
            self.register_skill(Skill(
                id=skill_id,
                name=name,
                content=content,
                path=path,
            ))
    
    def _get_skill_metadata(self) -> list[SkillMetadata]:
        """Convert skills to metadata for selector."""
        metadata = []
        
        for skill_id, skill in self.skills.items():
            quality = self.tracker.get_skill_quality(skill_id)
            stats = self.tracker.stats.get(skill_id)
            
            metadata.append(SkillMetadata(
                id=skill_id,
                name=skill.name,
                description=skill.content[:200],
                embedding=skill.embedding,
                success_rate=quality,
                usage_count=stats.total_uses if stats else 0,
                token_count=skill.token_count,
            ))
        
        return metadata
    
    def select_skills(self, task: str) -> tuple[list[str], list[float], SelectionState]:
        """
        Select skills for a task using RL policy.
        
        Returns:
            Tuple of (skill_ids, confidence_scores, state)
        """
        if not self.skills:
            return [], [], None
        
        # Embed task
        task_embedding = self.embedder.embed_task(task)
        
        # Build state
        state = SelectionState(
            task_embedding=task_embedding,
            available_skills=self._get_skill_metadata(),
            context_budget=self.config.context_budget,
        )
        
        # Select with policy
        if self.config.use_rl_selection:
            action = self.policy.select(
                state,
                temperature=self.config.temperature,
                deterministic=False,
            )
            return action.selected_skill_ids, action.confidence_scores, state
        else:
            # Fallback: similarity-based selection
            similarities = self.embedder.find_relevant_skills(
                task_embedding,
                {s.id: s.embedding for s in self.skills.values() if s.embedding is not None},
                top_k=self.config.max_skills,
            )
            return [s[0] for s in similarities], [s[1] for s in similarities], state
    
    def build_skill_context(self, skill_ids: list[str]) -> str:
        """Build context string from selected skills."""
        if not skill_ids:
            return ""
        
        parts = ["# Loaded Skills\n"]
        for skill_id in skill_ids:
            if skill_id in self.skills:
                skill = self.skills[skill_id]
                parts.append(f"\n## {skill.name}\n{skill.content}\n")
        
        return "\n".join(parts)
    
    async def run_task(self, task: str) -> TaskResult:
        """
        Run a task with RL-selected skills.
        
        Args:
            task: Task description
            
        Returns:
            Task result with outcome
        """
        task_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        try:
            # Select skills
            selected_ids, confidence_scores, state = self.select_skills(task)
            
            # Build skill context
            skill_context = self.build_skill_context(selected_ids)
            
            # Create or update agent
            if not self._agent_created:
                agent_state = self.client.agents.create(
                    model=self.config.model,
                    memory_blocks=[
                        {
                            "label": "skills",
                            "value": skill_context or "No skills loaded.",
                        },
                        {
                            "label": "persona",
                            "value": "I am a helpful AI assistant with access to specialized skills.",
                        },
                    ],
                )
                self._agent_id = agent_state.id
                self._agent_created = True
            else:
                # Update skills memory block
                self.client.agents.memory.update(
                    agent_id=self._agent_id,
                    block_label="skills",
                    value=skill_context or "No skills loaded.",
                )
            
            # Execute task
            response = self.client.agents.messages.create(
                agent_id=self._agent_id,
                input=task,
            )
            
            # Extract response text
            response_text = ""
            for msg in response.messages:
                if hasattr(msg, "text"):
                    response_text += msg.text
                elif hasattr(msg, "content"):
                    response_text += str(msg.content)
            
            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return TaskResult(
                task_id=task_id,
                success=True,
                response=response_text,
                skills_used=selected_ids,
                confidence_scores=confidence_scores,
                tokens_used=len(response_text.split()) * 2,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task_id,
                success=False,
                response="",
                skills_used=[],
                confidence_scores=[],
                tokens_used=0,
                error=str(e),
            )
    
    def run_task_sync(self, task: str) -> TaskResult:
        """Synchronous wrapper for run_task."""
        return asyncio.run(self.run_task(task))
    
    def run_task_mock(self, task: str) -> TaskResult:
        """
        Run task without Letta API (for testing).
        
        Simulates task execution and returns mock result.
        """
        task_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        
        # Select skills
        selected_ids, confidence_scores, state = self.select_skills(task)
        
        # Mock response
        skill_names = [self.skills[sid].name for sid in selected_ids if sid in self.skills]
        response = f"[Mock] Task: {task}\nSkills used: {', '.join(skill_names) or 'None'}"
        
        latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return TaskResult(
            task_id=task_id,
            success=True,
            response=response,
            skills_used=selected_ids,
            confidence_scores=confidence_scores,
            tokens_used=len(response.split()) * 2,
            latency_ms=latency_ms,
        )
    
    def record_feedback(
        self,
        result: TaskResult,
        success: bool,
        reward: Optional[float] = None,
    ):
        """
        Record task outcome for learning.
        
        Args:
            result: The task result to provide feedback on
            success: Whether the task was successful
            reward: Optional custom reward (default: 1.0 for success, -0.5 for failure)
        """
        if reward is None:
            reward = 1.0 if success else -0.5
        
        # Record in tracker
        outcome = TaskOutcome(
            task_id=result.task_id,
            skills_used=result.skills_used,
            success=success,
            reward=reward,
            timestamp=datetime.now(),
        )
        self.tracker.record_outcome(outcome)
        
        # Store experience for policy training
        # Need to reconstruct state from task
        # In production, store state along with result
    
    def train(self, min_samples: int = 32) -> dict:
        """
        Train policy on collected experiences.
        
        Returns:
            Training metrics
        """
        if len(self.policy.experiences) < min_samples:
            return {
                "status": "insufficient_data",
                "samples": len(self.policy.experiences),
                "required": min_samples,
            }
        
        metrics = self.policy.train()
        return metrics
    
    def save(self, path: str):
        """Save agent state (policy + tracker)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        self.policy.save(str(path / "policy"))
        self.tracker.save(str(path / "tracker.json"))
    
    def load(self, path: str):
        """Load agent state."""
        path = Path(path)
        
        policy_path = path / "policy"
        if policy_path.exists():
            self.policy.load(str(policy_path))
        
        tracker_path = path / "tracker.json"
        if tracker_path.exists():
            self.tracker.load(str(tracker_path))
    
    def get_stats(self) -> dict:
        """Get agent statistics."""
        report = self.tracker.get_skill_report()
        
        return {
            "total_skills": len(self.skills),
            "total_outcomes": report["total_outcomes"],
            "avg_success_rate": report["avg_success_rate"],
            "top_skills": report["top_skills"],
            "policy_training_step": self.policy.training_step,
            "experiences_buffered": len(self.policy.experiences),
        }
