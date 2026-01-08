"""
Task Execution Routes
"""

from fastapi import APIRouter, HTTPException
from ..schemas import TaskRequest, TaskResponse
from ..server import get_state

router = APIRouter()


@router.post("/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task with RL-selected skills."""
    executor = get_state("executor")
    if not executor:
        raise HTTPException(500, "Executor not initialized")
    
    result = executor.execute(
        task_description=request.description,
        agent_id=request.agent_id,
        temperature=request.temperature,
    )
    
    return TaskResponse(
        task_id=result.task_id,
        success=result.success,
        response=result.response,
        error=result.error,
        skills_used=result.skills_used or [],
        reward=result.reward,
        tokens_used=result.tokens_used,
        latency_ms=result.latency_ms,
    )


@router.post("/select")
async def select_skills(request: TaskRequest):
    """Select skills without executing."""
    selector = get_state("selector")
    if not selector:
        raise HTTPException(500, "Selector not initialized")
    
    selected, probs, _ = selector.select(
        task_description=request.description,
        temperature=request.temperature,
    )
    
    return {
        "skills": selected,
        "probabilities": probs,
    }
