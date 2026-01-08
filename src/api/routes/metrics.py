"""
Metrics Routes
"""

from fastapi import APIRouter, HTTPException
from ..schemas import DashboardResponse
from ..server import get_state

router = APIRouter()


@router.get("/dashboard")
async def dashboard():
    """Get dashboard data."""
    analytics = get_state("analytics")
    selector = get_state("selector")
    
    if not analytics:
        raise HTTPException(500, "Analytics not initialized")
    
    data = analytics.get_dashboard_data()
    
    # Add policy info
    if selector:
        data["summary"]["policy_version"] = selector.policy.training_step
        data["summary"]["buffer_size"] = len(selector.trainer.buffer)
    
    return data


@router.get("/summary")
async def summary(days: int = 30):
    """Get summary statistics."""
    analytics = get_state("analytics")
    if not analytics:
        raise HTTPException(500, "Analytics not initialized")
    
    stats = analytics.get_usage_stats(days)
    
    return {
        "total_tasks": stats.total_tasks,
        "success_rate": stats.success_rate,
        "avg_reward": stats.avg_reward,
        "unique_skills": stats.unique_skills_used,
        "avg_skills_per_task": stats.avg_skills_per_task,
    }


@router.get("/skills")
async def skill_metrics(days: int = 30):
    """Get per-skill metrics."""
    analytics = get_state("analytics")
    if not analytics:
        raise HTTPException(500, "Analytics not initialized")
    
    usage = analytics.get_skill_usage(days)
    
    return {
        "skills": [
            {
                "skill_id": s.skill_id,
                "skill_name": s.skill_name,
                "total_uses": s.total_uses,
                "success_rate": s.success_rate,
                "avg_reward": s.avg_reward,
                "co_used_with": s.co_used_with,
            }
            for s in usage
        ]
    }


@router.get("/trends/success")
async def success_trend(days: int = 14):
    """Get success rate trend."""
    analytics = get_state("analytics")
    if not analytics:
        raise HTTPException(500, "Analytics not initialized")
    
    trend = analytics.get_success_trend(days)
    
    return {
        "trend": [
            {"date": p.timestamp.isoformat(), "value": p.value, "count": p.count}
            for p in trend
        ]
    }


@router.get("/trends/reward")
async def reward_trend(days: int = 14):
    """Get reward trend."""
    analytics = get_state("analytics")
    if not analytics:
        raise HTTPException(500, "Analytics not initialized")
    
    trend = analytics.get_reward_trend(days)
    
    return {
        "trend": [
            {"date": p.timestamp.isoformat(), "value": p.value, "count": p.count}
            for p in trend
        ]
    }


@router.get("/compare")
async def compare_policies(v1: int, v2: int):
    """Compare two policy versions."""
    analytics = get_state("analytics")
    if not analytics:
        raise HTTPException(500, "Analytics not initialized")
    
    return analytics.compare_policies(v1, v2)
