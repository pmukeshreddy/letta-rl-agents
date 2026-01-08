"""
Skill Management Routes
"""

from fastapi import APIRouter, HTTPException
from ..schemas import SkillCreate, SkillUpdate, SkillResponse
from ..server import get_state

router = APIRouter()


@router.get("/")
async def list_skills(category: str = None, limit: int = 100):
    """List all skills."""
    repo = get_state("skill_repo")
    if not repo:
        raise HTTPException(500, "Repository not initialized")
    
    if category:
        skills = repo.list_by_category(category)
    else:
        skills = repo.list_all()
    
    return {
        "skills": [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "category": s.category,
                "token_count": s.token_count,
                "success_rate": s.success_rate,
                "total_uses": s.total_uses,
            }
            for s in skills[:limit]
        ],
        "total": len(skills),
    }


@router.get("/{skill_id}")
async def get_skill(skill_id: str):
    """Get skill by ID."""
    repo = get_state("skill_repo")
    if not repo:
        raise HTTPException(500, "Repository not initialized")
    
    skill = repo.get(skill_id)
    if not skill:
        raise HTTPException(404, f"Skill '{skill_id}' not found")
    
    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "content": skill.content,
        "category": skill.category,
        "tags": skill.tags,
        "token_count": skill.token_count,
        "success_rate": skill.success_rate,
        "avg_reward": skill.avg_reward,
        "total_uses": skill.total_uses,
    }


@router.post("/", response_model=SkillResponse)
async def create_skill(skill: SkillCreate):
    """Create a new skill."""
    repo = get_state("skill_repo")
    selector = get_state("selector")
    
    if not repo:
        raise HTTPException(500, "Repository not initialized")
    
    # Check if exists
    if repo.get(skill.id):
        raise HTTPException(400, f"Skill '{skill.id}' already exists")
    
    created = repo.create(
        id=skill.id,
        name=skill.name,
        content=skill.content,
        description=skill.description,
        category=skill.category,
        tags=skill.tags,
    )
    
    # Register with selector
    if selector:
        from ...agents.executor import Skill as SkillData
        selector.register_skill(SkillData(
            id=created.id,
            name=created.name,
            description=created.description or "",
            content=created.content,
            category=created.category,
            token_count=created.token_count,
        ))
    
    return SkillResponse(
        id=created.id,
        name=created.name,
        description=created.description,
        category=created.category,
        tags=created.tags or [],
        token_count=created.token_count,
        success_rate=0.0,
        avg_reward=0.0,
        total_uses=0,
    )


@router.put("/{skill_id}")
async def update_skill(skill_id: str, update: SkillUpdate):
    """Update a skill."""
    repo = get_state("skill_repo")
    if not repo:
        raise HTTPException(500, "Repository not initialized")
    
    skill = repo.get(skill_id)
    if not skill:
        raise HTTPException(404, f"Skill '{skill_id}' not found")
    
    updates = update.model_dump(exclude_unset=True)
    updated = repo.update(skill_id, **updates)
    
    return {"status": "updated", "skill_id": skill_id}


@router.delete("/{skill_id}")
async def delete_skill(skill_id: str):
    """Delete a skill."""
    repo = get_state("skill_repo")
    if not repo:
        raise HTTPException(500, "Repository not initialized")
    
    if not repo.delete(skill_id):
        raise HTTPException(404, f"Skill '{skill_id}' not found")
    
    return {"status": "deleted", "skill_id": skill_id}


@router.get("/top/performers")
async def top_skills(limit: int = 10, min_uses: int = 5):
    """Get top performing skills."""
    repo = get_state("skill_repo")
    if not repo:
        raise HTTPException(500, "Repository not initialized")
    
    skills = repo.get_top_performers(limit, min_uses)
    
    return {
        "skills": [
            {
                "id": s.id,
                "name": s.name,
                "success_rate": s.success_rate,
                "total_uses": s.total_uses,
            }
            for s in skills
        ]
    }
