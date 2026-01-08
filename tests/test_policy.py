"""
Tests for the RL Skill Selector Policy.
"""

import pytest
import numpy as np
from src.selector.policy import (
    SkillSelectorPolicy,
    SkillMetadata,
    SelectionState,
    SelectionAction,
)


@pytest.fixture
def policy():
    """Create a fresh policy for testing."""
    return SkillSelectorPolicy(
        embedding_dim=64,  # Smaller for tests
        hidden_dim=32,
        max_skills=3,
    )


@pytest.fixture
def sample_skills():
    """Create sample skills with embeddings."""
    np.random.seed(42)
    return [
        SkillMetadata(
            id=f"skill-{i}",
            name=f"Skill {i}",
            description=f"Description for skill {i}",
            embedding=np.random.randn(64).astype(np.float32),
            success_rate=0.5 + i * 0.1,
            usage_count=10 * i,
            token_count=100 * (i + 1),
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_state(sample_skills):
    """Create a sample selection state."""
    np.random.seed(42)
    return SelectionState(
        task_embedding=np.random.randn(64).astype(np.float32),
        available_skills=sample_skills,
        context_budget=500,
    )


class TestSkillSelectorPolicy:
    """Tests for SkillSelectorPolicy."""
    
    def test_init(self, policy):
        """Test policy initialization."""
        assert policy.embedding_dim == 64
        assert policy.hidden_dim == 32
        assert policy.max_skills == 3
        assert policy.training_step == 0
        assert len(policy.experiences) == 0
    
    def test_select_returns_action(self, policy, sample_state):
        """Test that select returns a valid action."""
        action = policy.select(sample_state)
        
        assert isinstance(action, SelectionAction)
        assert len(action.selected_skill_ids) <= policy.max_skills
        assert len(action.selected_skill_ids) == len(action.confidence_scores)
        assert len(action.selected_skill_ids) == len(action.log_probs)
    
    def test_select_empty_skills(self, policy):
        """Test selection with no available skills."""
        state = SelectionState(
            task_embedding=np.random.randn(64).astype(np.float32),
            available_skills=[],
            context_budget=500,
        )
        
        action = policy.select(state)
        
        assert action.selected_skill_ids == []
        assert action.confidence_scores == []
    
    def test_select_deterministic(self, policy, sample_state):
        """Test deterministic selection is consistent."""
        action1 = policy.select(sample_state, deterministic=True)
        action2 = policy.select(sample_state, deterministic=True)
        
        assert action1.selected_skill_ids == action2.selected_skill_ids
    
    def test_select_respects_context_budget(self, policy, sample_skills):
        """Test that selection respects context budget."""
        # Set a very small budget
        state = SelectionState(
            task_embedding=np.random.randn(64).astype(np.float32),
            available_skills=sample_skills,
            context_budget=150,  # Only skill-0 (100 tokens) fits
        )
        
        action = policy.select(state, deterministic=True)
        
        # Should only select skills that fit
        for skill_id in action.selected_skill_ids:
            skill = next(s for s in sample_skills if s.id == skill_id)
            assert skill.token_count <= 150
    
    def test_confidence_scores_sum_valid(self, policy, sample_state):
        """Test that confidence scores are valid probabilities."""
        action = policy.select(sample_state)
        
        for score in action.confidence_scores:
            assert 0.0 <= score <= 1.0
    
    def test_estimate_value(self, policy, sample_state):
        """Test value estimation."""
        value = policy.estimate_value(sample_state)
        
        assert isinstance(value, float)
    
    def test_store_experience(self, policy, sample_state):
        """Test experience storage."""
        action = policy.select(sample_state)
        
        policy.store_experience(sample_state, action, reward=1.0)
        
        assert len(policy.experiences) == 1
        assert policy.experiences[0].reward == 1.0
    
    def test_train_insufficient_data(self, policy, sample_state):
        """Test training with insufficient data."""
        action = policy.select(sample_state)
        policy.store_experience(sample_state, action, reward=1.0)
        
        metrics = policy.train(minibatch_size=32)
        
        assert metrics["status"] == "insufficient_data"
    
    def test_train_success(self, policy, sample_state):
        """Test successful training."""
        # Collect enough experiences
        for _ in range(50):
            action = policy.select(sample_state)
            policy.store_experience(sample_state, action, reward=np.random.randn())
        
        initial_step = policy.training_step
        metrics = policy.train(minibatch_size=16)
        
        assert policy.training_step == initial_step + 1
        assert metrics["num_updates"] > 0
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
    
    def test_save_load(self, policy, sample_state, tmp_path):
        """Test saving and loading policy."""
        # Train a bit
        for _ in range(50):
            action = policy.select(sample_state)
            policy.store_experience(sample_state, action, reward=1.0)
        policy.train(minibatch_size=16)
        
        # Save
        save_path = str(tmp_path / "policy")
        policy.save(save_path)
        
        # Create new policy and load
        new_policy = SkillSelectorPolicy(embedding_dim=64, hidden_dim=32)
        new_policy.load(save_path)
        
        assert new_policy.training_step == policy.training_step


class TestSelectionState:
    """Tests for SelectionState."""
    
    def test_create_state(self, sample_skills):
        """Test creating a selection state."""
        state = SelectionState(
            task_embedding=np.zeros(64),
            available_skills=sample_skills,
            context_budget=1000,
        )
        
        assert len(state.available_skills) == 5
        assert state.context_budget == 1000


class TestSkillMetadata:
    """Tests for SkillMetadata."""
    
    def test_create_metadata(self):
        """Test creating skill metadata."""
        metadata = SkillMetadata(
            id="test-skill",
            name="Test Skill",
            description="A test skill",
            embedding=np.zeros(64),
            success_rate=0.8,
            usage_count=100,
            token_count=500,
        )
        
        assert metadata.id == "test-skill"
        assert metadata.success_rate == 0.8
        assert metadata.token_count == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
