"""
Tests for the selector module.
"""

import pytest
import numpy as np

from src.selector.policy import SkillSelectorPolicy, SelectionState, SkillMetadata
from src.selector.embeddings import EmbeddingModel, SkillEmbedder
from src.selector.buffer import ExperienceBuffer, ExperienceData
from src.selector.trainer import PPOTrainer, TrainingConfig


class TestSkillSelectorPolicy:
    """Tests for SkillSelectorPolicy."""
    
    def test_init(self):
        """Test policy initialization."""
        policy = SkillSelectorPolicy(embedding_dim=384, hidden_dim=256)
        assert policy.embedding_dim == 384
        assert policy.hidden_dim == 256
    
    def test_select_empty_skills(self):
        """Test selection with no skills."""
        policy = SkillSelectorPolicy()
        state = SelectionState(
            task_embedding=np.random.randn(384),
            available_skills=[],
        )
        action = policy.select(state)
        assert action.selected_skill_ids == []
    
    def test_select_with_skills(self):
        """Test selection with available skills."""
        policy = SkillSelectorPolicy()
        
        skills = [
            SkillMetadata(
                id=f"skill-{i}",
                name=f"Skill {i}",
                description="",
                embedding=np.random.randn(384),
                token_count=100,
            )
            for i in range(5)
        ]
        
        state = SelectionState(
            task_embedding=np.random.randn(384),
            available_skills=skills,
            context_budget=1000,
        )
        
        action = policy.select(state)
        assert len(action.selected_skill_ids) > 0
        assert len(action.selected_skill_ids) <= policy.max_skills
    
    def test_deterministic_selection(self):
        """Test deterministic selection."""
        policy = SkillSelectorPolicy()
        np.random.seed(42)
        
        skills = [
            SkillMetadata(
                id=f"skill-{i}",
                name=f"Skill {i}",
                description="",
                embedding=np.random.randn(384),
                token_count=100,
            )
            for i in range(3)
        ]
        
        state = SelectionState(
            task_embedding=np.random.randn(384),
            available_skills=skills,
        )
        
        action1 = policy.select(state, deterministic=True)
        action2 = policy.select(state, deterministic=True)
        
        assert action1.selected_skill_ids == action2.selected_skill_ids


class TestEmbeddingModel:
    """Tests for EmbeddingModel."""
    
    def test_mock_embedding(self):
        """Test mock embedding generation."""
        model = EmbeddingModel(use_cache=False)
        
        embedding = model.embed("test text")
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
    
    def test_deterministic_mock(self):
        """Test mock embeddings are deterministic."""
        model = EmbeddingModel(use_cache=False)
        
        e1 = model.embed("same text")
        e2 = model.embed("same text")
        
        np.testing.assert_array_equal(e1, e2)
    
    def test_different_texts(self):
        """Test different texts get different embeddings."""
        model = EmbeddingModel(use_cache=False)
        
        e1 = model.embed("text one")
        e2 = model.embed("text two")
        
        assert not np.allclose(e1, e2)


class TestExperienceBuffer:
    """Tests for ExperienceBuffer."""
    
    def test_add_experience(self):
        """Test adding experiences."""
        buffer = ExperienceBuffer(max_size=100)
        
        exp = ExperienceData(
            task_embedding=np.random.randn(384),
            available_skill_ids=["s1", "s2"],
            skill_embeddings={},
            skill_qualities={},
            context_budget=1000,
            selected_skill_ids=["s1"],
            action_probs={"s1": 0.8},
            action_log_probs={"s1": -0.2},
            reward=1.0,
        )
        
        buffer.add(exp)
        assert len(buffer) == 1
    
    def test_max_size(self):
        """Test buffer respects max size."""
        buffer = ExperienceBuffer(max_size=5)
        
        for i in range(10):
            exp = ExperienceData(
                task_embedding=np.random.randn(384),
                available_skill_ids=["s1"],
                skill_embeddings={},
                skill_qualities={},
                context_budget=1000,
                selected_skill_ids=["s1"],
                action_probs={"s1": 0.8},
                action_log_probs={"s1": -0.2},
                reward=float(i),
            )
            buffer.add(exp)
        
        assert len(buffer) == 5


class TestPPOTrainer:
    """Tests for PPOTrainer."""
    
    def test_init(self):
        """Test trainer initialization."""
        policy = SkillSelectorPolicy()
        trainer = PPOTrainer(policy)
        
        assert trainer.policy is policy
        assert trainer.total_updates == 0
    
    def test_train_insufficient_data(self):
        """Test training with insufficient data."""
        policy = SkillSelectorPolicy()
        config = TrainingConfig(minibatch_size=32)
        trainer = PPOTrainer(policy, config)
        
        # Add only a few experiences
        for _ in range(5):
            exp = ExperienceData(
                task_embedding=np.random.randn(384),
                available_skill_ids=["s1"],
                skill_embeddings={"s1": np.random.randn(384)},
                skill_qualities={"s1": 0.5},
                context_budget=1000,
                selected_skill_ids=["s1"],
                action_probs={"s1": 0.8},
                action_log_probs={"s1": -0.2},
                reward=1.0,
                value=0.5,
            )
            trainer.buffer.add(exp)
        
        trainer.buffer.end_episode()
        result = trainer.train()
        
        assert result is None  # Not enough data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
