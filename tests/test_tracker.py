"""
Tests for the Skill Quality Tracker.
"""

import pytest
from datetime import datetime, timedelta
from src.tracker.quality import SkillTracker, SkillStats, TaskOutcome


@pytest.fixture
def tracker():
    """Create a fresh tracker for testing."""
    return SkillTracker()


@pytest.fixture
def sample_outcome():
    """Create a sample task outcome."""
    return TaskOutcome(
        task_id="task-001",
        skills_used=["skill-a", "skill-b"],
        success=True,
        reward=1.0,
        timestamp=datetime.now(),
    )


class TestSkillTracker:
    """Tests for SkillTracker."""
    
    def test_init(self, tracker):
        """Test tracker initialization."""
        assert len(tracker.stats) == 0
        assert len(tracker.outcomes) == 0
    
    def test_record_outcome_creates_stats(self, tracker, sample_outcome):
        """Test that recording creates stats for new skills."""
        tracker.record_outcome(sample_outcome)
        
        assert "skill-a" in tracker.stats
        assert "skill-b" in tracker.stats
    
    def test_record_outcome_updates_stats(self, tracker, sample_outcome):
        """Test that recording updates existing stats."""
        tracker.record_outcome(sample_outcome)
        tracker.record_outcome(sample_outcome)
        
        assert tracker.stats["skill-a"].total_uses == 2
        assert tracker.stats["skill-a"].successful_uses == 2
    
    def test_record_failure(self, tracker):
        """Test recording a failed outcome."""
        outcome = TaskOutcome(
            task_id="task-002",
            skills_used=["skill-c"],
            success=False,
            reward=-0.5,
        )
        tracker.record_outcome(outcome)
        
        stats = tracker.stats["skill-c"]
        assert stats.total_uses == 1
        assert stats.successful_uses == 0
        assert stats.total_reward == -0.5
    
    def test_success_rate(self, tracker):
        """Test success rate calculation."""
        # Record 2 successes and 1 failure
        for success in [True, True, False]:
            outcome = TaskOutcome(
                task_id=f"task-{success}",
                skills_used=["skill-x"],
                success=success,
                reward=1.0 if success else -0.5,
            )
            tracker.record_outcome(outcome)
        
        assert tracker.stats["skill-x"].success_rate == pytest.approx(2/3)
    
    def test_get_skill_quality_unknown(self, tracker):
        """Test quality score for unknown skill."""
        quality = tracker.get_skill_quality("unknown-skill")
        assert quality == 0.5  # Neutral score for unknown
    
    def test_get_skill_quality_known(self, tracker, sample_outcome):
        """Test quality score for known skill."""
        tracker.record_outcome(sample_outcome)
        
        quality = tracker.get_skill_quality("skill-a")
        assert 0.0 <= quality <= 1.0
    
    def test_get_top_skills(self, tracker):
        """Test getting top skills."""
        # Create skills with different success rates
        for i in range(5):
            for j in range(5):  # 5 uses each
                outcome = TaskOutcome(
                    task_id=f"task-{i}-{j}",
                    skills_used=[f"skill-{i}"],
                    success=(j < i),  # skill-0: 0%, skill-1: 20%, etc.
                    reward=1.0,
                )
                tracker.record_outcome(outcome)
        
        top = tracker.get_top_skills(n=3, min_uses=3)
        
        assert len(top) == 3
        # Should be ordered by success rate
        assert top[0][1] >= top[1][1] >= top[2][1]
    
    def test_get_bad_skills(self, tracker):
        """Test identifying bad skills."""
        # Create a bad skill (low success rate)
        for _ in range(10):
            outcome = TaskOutcome(
                task_id="bad-task",
                skills_used=["bad-skill"],
                success=False,
                reward=-0.5,
            )
            tracker.record_outcome(outcome)
        
        bad = tracker.get_bad_skills(threshold=0.3, min_uses=5)
        assert "bad-skill" in bad
    
    def test_get_skill_report(self, tracker, sample_outcome):
        """Test skill report generation."""
        tracker.record_outcome(sample_outcome)
        
        report = tracker.get_skill_report()
        
        assert "total_skills_tracked" in report
        assert "total_outcomes" in report
        assert "top_skills" in report
        assert report["total_outcomes"] == 1
    
    def test_save_load(self, tracker, sample_outcome, tmp_path):
        """Test saving and loading tracker state."""
        tracker.record_outcome(sample_outcome)
        
        save_path = tmp_path / "tracker.json"
        tracker.save(str(save_path))
        
        new_tracker = SkillTracker()
        new_tracker.load(str(save_path))
        
        assert "skill-a" in new_tracker.stats
        assert new_tracker.stats["skill-a"].total_uses == 1


class TestSkillStats:
    """Tests for SkillStats."""
    
    def test_success_rate_empty(self):
        """Test success rate with no uses."""
        stats = SkillStats(skill_id="test")
        assert stats.success_rate == 0.0
    
    def test_success_rate_calculated(self):
        """Test success rate calculation."""
        stats = SkillStats(
            skill_id="test",
            total_uses=10,
            successful_uses=7,
        )
        assert stats.success_rate == 0.7
    
    def test_avg_reward(self):
        """Test average reward calculation."""
        stats = SkillStats(
            skill_id="test",
            total_uses=5,
            total_reward=2.5,
        )
        assert stats.avg_reward == 0.5


class TestTaskOutcome:
    """Tests for TaskOutcome."""
    
    def test_create_outcome(self):
        """Test creating a task outcome."""
        outcome = TaskOutcome(
            task_id="test-task",
            skills_used=["skill-1"],
            success=True,
            reward=1.0,
        )
        
        assert outcome.task_id == "test-task"
        assert outcome.success is True
        assert isinstance(outcome.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
