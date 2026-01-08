"""
Skill Repository

CRUD operations for skills with database persistence.
"""

from typing import Optional
from datetime import datetime
import numpy as np

from ..db.models import Database, Skill, SkillStats


class SkillRepository:
    """
    Skill storage and retrieval.
    
    Wraps database operations with business logic.
    """
    
    def __init__(self, db: Database):
        self.db = db
    
    def create(
        self,
        id: str,
        name: str,
        content: str,
        description: str = "",
        category: str = "general",
        tags: list[str] = None,
        embedding: Optional[bytes] = None,
        token_count: int = 0,
        file_path: Optional[str] = None,
    ) -> Skill:
        """Create a new skill."""
        with self.db.get_session() as session:
            skill = Skill(
                id=id,
                name=name,
                content=content,
                description=description,
                category=category,
                tags=tags or [],
                embedding=embedding,
                token_count=token_count,
                file_path=file_path,
            )
            
            # Create associated stats
            stats = SkillStats(skill_id=id)
            skill.stats = stats
            
            session.add(skill)
            session.commit()
            session.refresh(skill)
            return skill
    
    def get(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID."""
        with self.db.get_session() as session:
            return session.query(Skill).filter(Skill.id == skill_id).first()
    
    def get_many(self, skill_ids: list[str]) -> list[Skill]:
        """Get multiple skills by IDs."""
        with self.db.get_session() as session:
            return session.query(Skill).filter(Skill.id.in_(skill_ids)).all()
    
    def list_all(self) -> list[Skill]:
        """List all skills."""
        with self.db.get_session() as session:
            return session.query(Skill).order_by(Skill.name).all()
    
    def list_by_category(self, category: str) -> list[Skill]:
        """List skills by category."""
        with self.db.get_session() as session:
            return (
                session.query(Skill)
                .filter(Skill.category == category)
                .order_by(Skill.name)
                .all()
            )
    
    def search(self, query: str, limit: int = 10) -> list[Skill]:
        """Search skills by name/description."""
        with self.db.get_session() as session:
            pattern = f"%{query}%"
            return (
                session.query(Skill)
                .filter(
                    (Skill.name.ilike(pattern)) |
                    (Skill.description.ilike(pattern))
                )
                .limit(limit)
                .all()
            )
    
    def update(
        self,
        skill_id: str,
        **kwargs,
    ) -> Optional[Skill]:
        """Update skill fields."""
        with self.db.get_session() as session:
            skill = session.query(Skill).filter(Skill.id == skill_id).first()
            if not skill:
                return None
            
            for key, value in kwargs.items():
                if hasattr(skill, key):
                    setattr(skill, key, value)
            
            skill.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(skill)
            return skill
    
    def update_stats(
        self,
        skill_id: str,
        success: bool,
        reward: float,
        task_type: Optional[str] = None,
    ):
        """Update skill statistics after use."""
        with self.db.get_session() as session:
            skill = session.query(Skill).filter(Skill.id == skill_id).first()
            if not skill:
                return
            
            # Update denormalized stats
            skill.total_uses += 1
            if success:
                skill.successful_uses += 1
            skill.total_reward += reward
            skill.updated_at = datetime.utcnow()
            
            # Update detailed stats
            if skill.stats:
                stats = skill.stats
                stats.total_uses += 1
                if success:
                    stats.successful_uses += 1
                else:
                    stats.failed_uses += 1
                stats.total_reward += reward
                stats.last_used_at = datetime.utcnow()
                
                if success:
                    stats.last_success_at = datetime.utcnow()
                else:
                    stats.last_failure_at = datetime.utcnow()
                
                # Track by task type
                if task_type:
                    type_stats = stats.task_type_stats or {}
                    if task_type not in type_stats:
                        type_stats[task_type] = {"uses": 0, "success": 0}
                    type_stats[task_type]["uses"] += 1
                    if success:
                        type_stats[task_type]["success"] += 1
                    stats.task_type_stats = type_stats
            
            session.commit()
    
    def update_embedding(self, skill_id: str, embedding: np.ndarray):
        """Update skill embedding."""
        with self.db.get_session() as session:
            skill = session.query(Skill).filter(Skill.id == skill_id).first()
            if skill:
                skill.embedding = embedding.tobytes()
                skill.updated_at = datetime.utcnow()
                session.commit()
    
    def delete(self, skill_id: str) -> bool:
        """Delete a skill."""
        with self.db.get_session() as session:
            skill = session.query(Skill).filter(Skill.id == skill_id).first()
            if skill:
                session.delete(skill)
                session.commit()
                return True
            return False
    
    def get_top_performers(self, limit: int = 10, min_uses: int = 5) -> list[Skill]:
        """Get top performing skills by success rate."""
        with self.db.get_session() as session:
            from sqlalchemy import Float
            return (
                session.query(Skill)
                .filter(Skill.total_uses >= min_uses)
                .order_by(
                    (Skill.successful_uses.cast(Float) / Skill.total_uses).desc()
                )
                .limit(limit)
                .all()
            )
    
    def get_underperformers(self, limit: int = 10, min_uses: int = 5) -> list[Skill]:
        """Get worst performing skills."""
        with self.db.get_session() as session:
            from sqlalchemy import Float
            return (
                session.query(Skill)
                .filter(Skill.total_uses >= min_uses)
                .order_by(
                    (Skill.successful_uses.cast(Float) / Skill.total_uses).asc()
                )
                .limit(limit)
                .all()
            )
    
    def get_unused(self, days: int = 7) -> list[Skill]:
        """Get skills not used recently."""
        with self.db.get_session() as session:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            return (
                session.query(Skill)
                .filter(
                    (Skill.updated_at < cutoff) |
                    (Skill.total_uses == 0)
                )
                .all()
            )
