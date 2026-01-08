"""
Usage Analytics

Analyze skill usage patterns and system performance.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

from ..db.models import Database, Skill, TaskOutcome, Experience, Policy


@dataclass
class UsageStats:
    """Overall usage statistics."""
    total_tasks: int
    successful_tasks: int
    total_skills_used: int
    unique_skills_used: int
    avg_skills_per_task: float
    avg_reward: float
    success_rate: float


@dataclass
class SkillUsageStats:
    """Per-skill usage statistics."""
    skill_id: str
    skill_name: str
    total_uses: int
    success_count: int
    success_rate: float
    avg_reward: float
    co_used_with: list[tuple[str, int]]  # (skill_id, count)


@dataclass
class TimeSeriesPoint:
    """Single point in time series."""
    timestamp: datetime
    value: float
    count: int = 0


class Analytics:
    """
    System analytics and reporting.
    
    Provides:
    - Usage statistics
    - Performance trends
    - Skill co-occurrence analysis
    - Policy comparison
    """
    
    def __init__(self, db: Database):
        self.db = db
    
    def get_usage_stats(self, days: int = 30) -> UsageStats:
        """Get overall usage statistics."""
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            outcomes = (
                session.query(TaskOutcome)
                .filter(TaskOutcome.created_at >= cutoff)
                .all()
            )
            
            if not outcomes:
                return UsageStats(
                    total_tasks=0,
                    successful_tasks=0,
                    total_skills_used=0,
                    unique_skills_used=0,
                    avg_skills_per_task=0,
                    avg_reward=0,
                    success_rate=0,
                )
            
            total = len(outcomes)
            successful = sum(1 for o in outcomes if o.success)
            
            all_skills = []
            for o in outcomes:
                all_skills.extend([s.id for s in o.skills])
            
            return UsageStats(
                total_tasks=total,
                successful_tasks=successful,
                total_skills_used=len(all_skills),
                unique_skills_used=len(set(all_skills)),
                avg_skills_per_task=len(all_skills) / total if total else 0,
                avg_reward=sum(o.reward for o in outcomes) / total,
                success_rate=successful / total,
            )
    
    def get_skill_usage(self, days: int = 30) -> list[SkillUsageStats]:
        """Get per-skill usage statistics."""
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            skills = session.query(Skill).all()
            
            results = []
            for skill in skills:
                # Get outcomes where this skill was used
                outcomes = (
                    session.query(TaskOutcome)
                    .join(TaskOutcome.skills)
                    .filter(Skill.id == skill.id)
                    .filter(TaskOutcome.created_at >= cutoff)
                    .all()
                )
                
                if not outcomes:
                    continue
                
                success_count = sum(1 for o in outcomes if o.success)
                
                # Co-occurrence analysis
                co_skills = defaultdict(int)
                for o in outcomes:
                    for s in o.skills:
                        if s.id != skill.id:
                            co_skills[s.id] += 1
                
                co_used = sorted(co_skills.items(), key=lambda x: x[1], reverse=True)[:5]
                
                results.append(SkillUsageStats(
                    skill_id=skill.id,
                    skill_name=skill.name,
                    total_uses=len(outcomes),
                    success_count=success_count,
                    success_rate=success_count / len(outcomes),
                    avg_reward=sum(o.reward for o in outcomes) / len(outcomes),
                    co_used_with=co_used,
                ))
            
            # Sort by usage
            results.sort(key=lambda x: x.total_uses, reverse=True)
            return results
    
    def get_success_trend(
        self,
        days: int = 30,
        interval: str = "day",
    ) -> list[TimeSeriesPoint]:
        """Get success rate over time."""
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            outcomes = (
                session.query(TaskOutcome)
                .filter(TaskOutcome.created_at >= cutoff)
                .order_by(TaskOutcome.created_at)
                .all()
            )
            
            if not outcomes:
                return []
            
            # Group by interval
            if interval == "day":
                key_fn = lambda o: o.created_at.date()
            elif interval == "hour":
                key_fn = lambda o: o.created_at.replace(minute=0, second=0, microsecond=0)
            else:
                key_fn = lambda o: o.created_at.date()
            
            grouped = defaultdict(list)
            for o in outcomes:
                grouped[key_fn(o)].append(o)
            
            points = []
            for key in sorted(grouped.keys()):
                outcomes_in_period = grouped[key]
                success_count = sum(1 for o in outcomes_in_period if o.success)
                points.append(TimeSeriesPoint(
                    timestamp=datetime.combine(key, datetime.min.time()) if isinstance(key, type(datetime.now().date())) else key,
                    value=success_count / len(outcomes_in_period),
                    count=len(outcomes_in_period),
                ))
            
            return points
    
    def get_reward_trend(self, days: int = 30) -> list[TimeSeriesPoint]:
        """Get average reward over time."""
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            outcomes = (
                session.query(TaskOutcome)
                .filter(TaskOutcome.created_at >= cutoff)
                .order_by(TaskOutcome.created_at)
                .all()
            )
            
            if not outcomes:
                return []
            
            grouped = defaultdict(list)
            for o in outcomes:
                grouped[o.created_at.date()].append(o)
            
            points = []
            for date in sorted(grouped.keys()):
                outcomes_on_date = grouped[date]
                avg_reward = sum(o.reward for o in outcomes_on_date) / len(outcomes_on_date)
                points.append(TimeSeriesPoint(
                    timestamp=datetime.combine(date, datetime.min.time()),
                    value=avg_reward,
                    count=len(outcomes_on_date),
                ))
            
            return points
    
    def compare_policies(
        self,
        policy_v1: int,
        policy_v2: int,
    ) -> dict:
        """Compare performance of two policy versions."""
        with self.db.get_session() as session:
            outcomes_v1 = (
                session.query(TaskOutcome)
                .filter(TaskOutcome.policy_version == policy_v1)
                .all()
            )
            
            outcomes_v2 = (
                session.query(TaskOutcome)
                .filter(TaskOutcome.policy_version == policy_v2)
                .all()
            )
            
            def compute_stats(outcomes):
                if not outcomes:
                    return {"count": 0, "success_rate": 0, "avg_reward": 0}
                return {
                    "count": len(outcomes),
                    "success_rate": sum(1 for o in outcomes if o.success) / len(outcomes),
                    "avg_reward": sum(o.reward for o in outcomes) / len(outcomes),
                }
            
            stats_v1 = compute_stats(outcomes_v1)
            stats_v2 = compute_stats(outcomes_v2)
            
            return {
                f"policy_v{policy_v1}": stats_v1,
                f"policy_v{policy_v2}": stats_v2,
                "improvement": {
                    "success_rate": stats_v2["success_rate"] - stats_v1["success_rate"],
                    "avg_reward": stats_v2["avg_reward"] - stats_v1["avg_reward"],
                },
            }
    
    def get_dashboard_data(self) -> dict:
        """Get all data needed for dashboard."""
        usage = self.get_usage_stats(days=30)
        skill_usage = self.get_skill_usage(days=30)
        success_trend = self.get_success_trend(days=14)
        
        return {
            "summary": {
                "total_tasks": usage.total_tasks,
                "success_rate": usage.success_rate,
                "avg_reward": usage.avg_reward,
                "unique_skills": usage.unique_skills_used,
            },
            "top_skills": [
                {
                    "id": s.skill_id,
                    "name": s.skill_name,
                    "uses": s.total_uses,
                    "success_rate": s.success_rate,
                }
                for s in skill_usage[:10]
            ],
            "success_trend": [
                {"date": p.timestamp.isoformat(), "rate": p.value, "count": p.count}
                for p in success_trend
            ],
        }
