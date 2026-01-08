"""
FastAPI Server

Main API server for the RL skill selector.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

from ..db.session import init_db
from ..db.models import Database
from ..agents.executor import SkillSelector, TaskExecutor
from ..agents.client import LettaClient, MockLettaClient
from ..skills.repository import SkillRepository
from ..tracker.analytics import Analytics

from .routes import tasks, skills, training, metrics


# Global state
state = {}


def get_state(key: str):
    """Get item from app state."""
    return state.get(key)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize app state on startup."""
    # Initialize database
    db_url = os.getenv("DATABASE_URL", "sqlite:///letta_rl.db")
    db = init_db(db_url)
    
    # Initialize components
    skill_repo = SkillRepository(db)
    selector = SkillSelector()
    
    # Use mock client if no API key
    if os.getenv("LETTA_API_KEY"):
        letta = LettaClient()
    else:
        letta = MockLettaClient()
    
    executor = TaskExecutor(selector, letta, skill_repo)
    analytics = Analytics(db)
    
    # Store in state
    state["db"] = db
    state["skill_repo"] = skill_repo
    state["selector"] = selector
    state["executor"] = executor
    state["analytics"] = analytics
    
    yield
    
    state.clear()


# Create app
app = FastAPI(
    title="Letta RL Skill Selector",
    description="RL-based skill selection for Letta agents",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tasks.router, prefix="/tasks", tags=["Tasks"])
app.include_router(skills.router, prefix="/skills", tags=["Skills"])
app.include_router(training.router, prefix="/training", tags=["Training"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])


@app.get("/health")
async def health():
    """Health check."""
    selector = state.get("selector")
    return {
        "status": "healthy",
        "database": state.get("db") is not None,
        "skills_loaded": len(selector._skill_metadata) if selector else 0,
        "policy_version": selector.policy.training_step if selector else 0,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {"name": "Letta RL Skill Selector", "docs": "/docs"}
