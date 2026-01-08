"""
Training Routes
"""

from fastapi import APIRouter, HTTPException
from ..schemas import TrainingRequest, TrainingResponse, PolicyResponse
from ..server import get_state

router = APIRouter()


@router.post("/run", response_model=TrainingResponse)
async def run_training(request: TrainingRequest = None):
    """Run a training update."""
    executor = get_state("executor")
    if not executor:
        raise HTTPException(500, "Executor not initialized")
    
    # Update config if provided
    if request:
        executor.selector.trainer.config.epochs_per_update = request.epochs
        executor.selector.trainer.config.minibatch_size = request.batch_size
        executor.selector.trainer.config.policy_lr = request.learning_rate
    
    # Run training
    metrics = executor.train()
    
    if metrics is None:
        return TrainingResponse(
            status="insufficient_data",
            policy_version=executor.selector.policy.training_step,
            num_samples=len(executor.selector.trainer.buffer),
        )
    
    return TrainingResponse(
        status="completed",
        policy_version=executor.selector.policy.training_step,
        policy_loss=metrics.get("policy_loss"),
        value_loss=metrics.get("value_loss"),
        entropy=metrics.get("entropy"),
        num_samples=metrics.get("num_samples", 0),
    )


@router.get("/status")
async def training_status():
    """Get training status."""
    selector = get_state("selector")
    if not selector:
        raise HTTPException(500, "Selector not initialized")
    
    buffer_stats = selector.trainer.buffer.get_stats() if hasattr(selector.trainer.buffer, 'get_stats') else {}
    
    return {
        "policy_version": selector.policy.training_step,
        "buffer_size": len(selector.trainer.buffer),
        "total_updates": selector.trainer.total_updates,
        "buffer_stats": buffer_stats,
    }


@router.get("/policies")
async def list_policies():
    """List all policy versions."""
    db = get_state("db")
    if not db:
        raise HTTPException(500, "Database not initialized")
    
    from ...db.models import PolicyRepository, Policy
    repo = PolicyRepository(db)
    policies = repo.list_all()
    
    return {
        "policies": [
            {
                "version": p.version,
                "is_active": p.is_active,
                "trained_on_experiences": p.trained_on_experiences,
                "eval_success_rate": p.eval_success_rate,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in policies
        ]
    }


@router.post("/policies/{version}/activate")
async def activate_policy(version: int):
    """Activate a specific policy version."""
    db = get_state("db")
    selector = get_state("selector")
    
    if not db:
        raise HTTPException(500, "Database not initialized")
    
    from ...db.models import PolicyRepository
    repo = PolicyRepository(db)
    
    if not repo.activate(version):
        raise HTTPException(404, f"Policy version {version} not found")
    
    # Load into selector
    policy = repo.get_by_version(version)
    if policy and selector:
        selector.policy.load(f"policies/v{version}")
    
    return {"status": "activated", "version": version}


@router.post("/save")
async def save_policy(name: str = None):
    """Save current policy to database."""
    selector = get_state("selector")
    db = get_state("db")
    
    if not selector or not db:
        raise HTTPException(500, "Not initialized")
    
    from ...db.models import PolicyRepository, Policy
    import pickle
    
    repo = PolicyRepository(db)
    version = repo.get_latest_version() + 1
    
    # Serialize weights
    weights = pickle.dumps({
        "task_encoder": selector.policy.task_encoder.parameters(),
        "skill_encoder": selector.policy.skill_encoder.parameters(),
        "attention": selector.policy.attention.parameters(),
        "value_head": selector.policy.value_head.parameters(),
    })
    
    policy = Policy(
        version=version,
        name=name or f"policy-v{version}",
        weights=weights,
        trained_on_experiences=selector.trainer.total_updates,
        is_active=True,
    )
    
    repo.create(policy)
    repo.activate(version)
    
    return {"status": "saved", "version": version}
