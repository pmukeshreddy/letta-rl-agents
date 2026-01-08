"""
API Schemas

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ============ Task Schemas ============

class TaskRequest(BaseModel):
    """Request to execute a task."""
    description: str = Field(..., description="Task description")
    agent_id: Optional[str] = Field(None, description="Specific agent to use")
    temperature: float = Field(1.0, ge=0, le=2, description="Selection temperature")


class TaskResponse(BaseModel):
    """Response from task execution."""
    task_id: str
    success: bool
    response: str
    error: Optional[str] = None
    skills_used: list[str]
    reward: float
    tokens_used: int
    latency_ms: int


# ============ Skill Schemas ============

class SkillCreate(BaseModel):
    """Create a new skill."""
    id: str = Field(..., pattern="^[a-z0-9-]+$")
    name: str
    content: str
    description: str = ""
    category: str = "general"
    tags: list[str] = []


class SkillUpdate(BaseModel):
    """Update a skill."""
    name: Optional[str] = None
    content: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None


class SkillResponse(BaseModel):
    """Skill details."""
    id: str
    name: str
    description: Optional[str]
    category: Optional[str]
    tags: Optional[list]
    token_count: int
    success_rate: float
    avg_reward: float
    total_uses: int


# ============ Training Schemas ============

class TrainingRequest(BaseModel):
    """Request to run training."""
    epochs: int = Field(4, ge=1, le=100)
    batch_size: int = Field(64, ge=8, le=512)
    learning_rate: float = Field(3e-4, ge=1e-6, le=1e-2)


class TrainingResponse(BaseModel):
    """Training run result."""
    status: str
    policy_version: int
    policy_loss: Optional[float] = None
    value_loss: Optional[float] = None
    entropy: Optional[float] = None
    num_samples: int = 0


class PolicyResponse(BaseModel):
    """Policy details."""
    version: int
    is_active: bool
    trained_on_experiences: int
    eval_success_rate: Optional[float]
    eval_avg_reward: Optional[float]


# ============ Metrics Schemas ============

class MetricsSummary(BaseModel):
    """Summary metrics."""
    total_tasks: int
    success_rate: float
    avg_reward: float
    unique_skills: int
    policy_version: int


class DashboardResponse(BaseModel):
    """Full dashboard data."""
    summary: dict
    top_skills: list[dict]
    success_trend: list[dict]


# ============ Health ============

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: bool
    skills_loaded: int
    policy_version: int
