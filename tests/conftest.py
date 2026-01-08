"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)


@pytest.fixture
def embedding_dim():
    """Default embedding dimension for tests."""
    return 64


@pytest.fixture
def sample_embedding(embedding_dim):
    """Generate a sample embedding."""
    return np.random.randn(embedding_dim).astype(np.float32)
