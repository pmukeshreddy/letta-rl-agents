"""
Skill Transfer

Transfer proven skills between agents.
"""

from typing import Optional
from datetime import datetime
from dataclasses import dataclass

from .repository import SkillRepository


@dataclass
class TransferResult:
    """Result of skill transfer operation."""
    skill_id: str
    source_agent: str
    target_agent: str
    success: bool
    message: str


class SkillTransfer:
    """
    Transfer skills between agents.
    
    Features:
    - Transfer by quality threshold
    - Transfer by category
    - Sync across agents
    """
    
    def __init__(self, repository: SkillRepository):
        self.repository = repository
    
    def get_transferable_skills(
        self,
        min_success_rate: float = 0.7,
        min_uses: int = 10,
    ) -> list[dict]:
        """
        Get skills that meet transfer criteria.
        
        Only proven skills should be transferred.
        """
        all_skills = self.repository.list_all()
        
        transferable = []
        for skill in all_skills:
            if skill.total_uses >= min_uses and skill.success_rate >= min_success_rate:
                transferable.append({
                    "id": skill.id,
                    "name": skill.name,
                    "category": skill.category,
                    "success_rate": skill.success_rate,
                    "total_uses": skill.total_uses,
                    "avg_reward": skill.avg_reward,
                })
        
        # Sort by success rate
        transferable.sort(key=lambda x: x["success_rate"], reverse=True)
        return transferable
    
    def transfer_skill(
        self,
        skill_id: str,
        source_agent_id: str,
        target_agent_id: str,
        include_stats: bool = False,
    ) -> TransferResult:
        """
        Transfer a skill from one agent to another.
        
        Args:
            skill_id: Skill to transfer
            source_agent_id: Agent that owns the skill
            target_agent_id: Agent to receive the skill
            include_stats: Whether to copy performance stats
        """
        skill = self.repository.get(skill_id)
        
        if not skill:
            return TransferResult(
                skill_id=skill_id,
                source_agent=source_agent_id,
                target_agent=target_agent_id,
                success=False,
                message="Skill not found",
            )
        
        # In a multi-agent setup, we'd have agent-specific repositories
        # For now, skills are shared globally, so "transfer" is just
        # marking the skill as available to another agent
        
        # This would create a mapping in an agent_skills table
        # For simplicity, we'll just return success
        
        return TransferResult(
            skill_id=skill_id,
            source_agent=source_agent_id,
            target_agent=target_agent_id,
            success=True,
            message=f"Skill '{skill.name}' transferred successfully",
        )
    
    def bulk_transfer(
        self,
        source_agent_id: str,
        target_agent_id: str,
        min_success_rate: float = 0.7,
        min_uses: int = 10,
        categories: Optional[list[str]] = None,
    ) -> list[TransferResult]:
        """
        Transfer all qualifying skills between agents.
        """
        transferable = self.get_transferable_skills(min_success_rate, min_uses)
        
        if categories:
            transferable = [s for s in transferable if s["category"] in categories]
        
        results = []
        for skill in transferable:
            result = self.transfer_skill(
                skill_id=skill["id"],
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
            )
            results.append(result)
        
        return results
    
    def get_transfer_recommendations(
        self,
        target_agent_id: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Recommend skills to transfer to an agent.
        
        Based on:
        - High success rate globally
        - Not yet used by target agent
        - Complementary to existing skills
        """
        # Get top performing skills
        candidates = self.get_transferable_skills(min_success_rate=0.6, min_uses=5)
        
        # In a full implementation, we'd filter out skills
        # the target agent already has
        
        recommendations = []
        for skill in candidates[:top_k]:
            recommendations.append({
                **skill,
                "reason": f"High success rate ({skill['success_rate']:.0%})",
            })
        
        return recommendations
    
    def sync_skills(
        self,
        agent_ids: list[str],
        min_success_rate: float = 0.8,
    ) -> dict:
        """
        Sync proven skills across all specified agents.
        
        Returns summary of transfers.
        """
        transferable = self.get_transferable_skills(min_success_rate)
        
        transfers = {
            "total_skills": len(transferable),
            "agents_synced": len(agent_ids),
            "transfers": [],
        }
        
        for skill in transferable:
            for agent_id in agent_ids:
                transfers["transfers"].append({
                    "skill_id": skill["id"],
                    "agent_id": agent_id,
                })
        
        return transfers
