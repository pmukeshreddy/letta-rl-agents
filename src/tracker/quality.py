"""
Skill Quality Tracker

Tracks which skills actually improve task success.
Maintains statistics and prunes ineffective skills.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class SkillStats:
    """Statistics for a single skill."""
    skill_id: str
    total_uses: int = 0
    successful_uses: int = 0
    total_reward: float = 0.0
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.successful_uses / self.total_uses
    
    @property
    def avg_reward(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.total_reward / self.total_uses


@dataclass
class TaskOutcome:
    """Outcome of a task execution."""
    task_id: str
    skills_used: list[str]
    success: bool
    reward: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


class SkillTracker:
    """
    Tracks skill quality and effectiveness.
    
    Features:
    - Per-skill success rates
    - Skill-task correlation
    - Automatic pruning of bad skills
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.stats: dict[str, SkillStats] = {}
        self.outcomes: list[TaskOutcome] = []
        self.db_path = db_path
        
        # Correlation tracking: which skills work for which task types
        self.skill_task_correlation: dict[str, dict[str, float]] = {}
        
    def record_outcome(self, outcome: TaskOutcome):
        """Record a task outcome and update skill stats."""
        self.outcomes.append(outcome)
        
        for skill_id in outcome.skills_used:
            if skill_id not in self.stats:
                self.stats[skill_id] = SkillStats(skill_id=skill_id)
            
            stats = self.stats[skill_id]
            stats.total_uses += 1
            stats.total_reward += outcome.reward
            stats.last_used = outcome.timestamp
            
            if outcome.success:
                stats.successful_uses += 1
                
    def get_skill_quality(self, skill_id: str) -> float:
        """
        Get quality score for a skill.
        
        Combines success rate with recency bonus.
        """
        if skill_id not in self.stats:
            return 0.5  # Unknown skill gets neutral score
            
        stats = self.stats[skill_id]
        
        # Base score from success rate
        score = stats.success_rate
        
        # Confidence adjustment (more uses = more confident)
        confidence = min(stats.total_uses / 10, 1.0)
        score = 0.5 + (score - 0.5) * confidence
        
        # Recency bonus
        if stats.last_used:
            days_ago = (datetime.now() - stats.last_used).days
            recency = max(0, 1 - days_ago / 30)  # Decay over 30 days
            score += recency * 0.1
            
        return min(max(score, 0.0), 1.0)
    
    def get_top_skills(
        self,
        n: int = 10,
        min_uses: int = 3,
    ) -> list[tuple[str, float]]:
        """Get top N skills by quality score."""
        qualified = [
            (sid, self.get_skill_quality(sid))
            for sid, stats in self.stats.items()
            if stats.total_uses >= min_uses
        ]
        qualified.sort(key=lambda x: x[1], reverse=True)
        return qualified[:n]
    
    def get_bad_skills(
        self,
        threshold: float = 0.3,
        min_uses: int = 5,
    ) -> list[str]:
        """Get skills that should be pruned."""
        return [
            sid
            for sid, stats in self.stats.items()
            if stats.total_uses >= min_uses and stats.success_rate < threshold
        ]
    
    def get_skill_report(self) -> dict:
        """Generate a report of skill effectiveness."""
        return {
            "total_skills_tracked": len(self.stats),
            "total_outcomes": len(self.outcomes),
            "top_skills": self.get_top_skills(5),
            "bad_skills": self.get_bad_skills(),
            "avg_success_rate": (
                sum(s.success_rate for s in self.stats.values()) / len(self.stats)
                if self.stats else 0.0
            ),
        }
    
    def save(self, path: str):
        """Save tracker state to file."""
        data = {
            "stats": {
                sid: {
                    "skill_id": s.skill_id,
                    "total_uses": s.total_uses,
                    "successful_uses": s.successful_uses,
                    "total_reward": s.total_reward,
                    "last_used": s.last_used.isoformat() if s.last_used else None,
                }
                for sid, s in self.stats.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """Load tracker state from file."""
        with open(path) as f:
            data = json.load(f)
        
        for sid, s in data["stats"].items():
            self.stats[sid] = SkillStats(
                skill_id=s["skill_id"],
                total_uses=s["total_uses"],
                successful_uses=s["successful_uses"],
                total_reward=s["total_reward"],
                last_used=(
                    datetime.fromisoformat(s["last_used"])
                    if s["last_used"] else None
                ),
            )
