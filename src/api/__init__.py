"""
API Module

FastAPI server and routes.
"""

from .server import app
from .schemas import (
    TaskRequest,
    TaskResponse,
    SkillCreate,
    SkillUpdate,
    SkillResponse,
    TrainingRequest,
    TrainingResponse,
    DashboardResponse,
)

__all__ = [
    "app",
    "TaskRequest",
    "TaskResponse",
    "SkillCreate",
    "SkillUpdate",
    "SkillResponse",
    "TrainingRequest",
    "TrainingResponse",
    "DashboardResponse",
]
