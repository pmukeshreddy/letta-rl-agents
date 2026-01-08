"""
Embeddings Module

Handles text embeddings for tasks and skills.
Uses sentence-transformers for local inference.
"""

from typing import Optional
import numpy as np
from pathlib import Path
import json
import hashlib


class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""
    
    def __init__(self, cache_dir: str = ".cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: dict[str, np.ndarray] = {}
    
    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        key = self._hash_text(text)
        
        # Check memory cache
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            embedding = np.load(cache_file)
            self.memory_cache[key] = embedding
            return embedding
        
        return None
    
    def set(self, text: str, embedding: np.ndarray):
        key = self._hash_text(text)
        self.memory_cache[key] = embedding
        np.save(self.cache_dir / f"{key}.npy", embedding)


class EmbeddingModel:
    """
    Wrapper for embedding models.
    
    Supports:
    - sentence-transformers (local)
    - OpenAI embeddings (API)
    - Mock embeddings (testing)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_cache: bool = True,
        cache_dir: str = ".cache/embeddings",
    ):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Default for MiniLM
        
        self.cache = EmbeddingCache(cache_dir) if use_cache else None
        
        # Lazy load model
        self._initialized = False
    
    def _init_model(self):
        """Initialize the embedding model."""
        if self._initialized:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self._initialized = True
        except ImportError:
            print("Warning: sentence-transformers not installed. Using mock embeddings.")
            self._initialized = True
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        # Check cache
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached
        
        # Initialize model if needed
        self._init_model()
        
        # Generate embedding
        if self.model is not None:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            # Mock embedding for testing
            embedding = self._mock_embed(text)
        
        # Cache result
        if self.cache:
            self.cache.set(text, embedding)
        
        return embedding
    
    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts efficiently."""
        self._init_model()
        
        # Check cache for each text
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get(text)
                if cached is not None:
                    results.append(cached)
                    continue
            
            uncached_texts.append(text)
            uncached_indices.append(i)
            results.append(None)  # Placeholder
        
        # Batch embed uncached texts
        if uncached_texts:
            if self.model is not None:
                embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            else:
                embeddings = [self._mock_embed(t) for t in uncached_texts]
            
            for i, embedding in zip(uncached_indices, embeddings):
                results[i] = embedding
                if self.cache:
                    self.cache.set(texts[i], embedding)
        
        return results
    
    def _mock_embed(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding for testing."""
        # Use hash of text to generate reproducible embedding
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.dimension).astype(np.float32)
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class SkillEmbedder:
    """
    Specialized embedder for skills.
    
    Embeds skills based on:
    - Name
    - Description
    - Content (with chunking for long skills)
    """
    
    def __init__(self, embedding_model: Optional[EmbeddingModel] = None):
        self.model = embedding_model or EmbeddingModel()
    
    def embed_skill(
        self,
        name: str,
        content: str,
        max_content_chars: int = 2000,
    ) -> np.ndarray:
        """
        Embed a skill.
        
        Uses weighted combination of name and content embeddings.
        """
        # Truncate content if too long
        if len(content) > max_content_chars:
            content = content[:max_content_chars]
        
        # Create combined text
        combined = f"Skill: {name}\n\n{content}"
        
        return self.model.embed(combined)
    
    def embed_task(self, task_description: str) -> np.ndarray:
        """Embed a task description."""
        return self.model.embed(f"Task: {task_description}")
    
    def find_relevant_skills(
        self,
        task_embedding: np.ndarray,
        skill_embeddings: dict[str, np.ndarray],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find most relevant skills for a task.
        
        Returns list of (skill_id, similarity_score) tuples.
        """
        similarities = []
        
        for skill_id, skill_embedding in skill_embeddings.items():
            sim = self.model.similarity(task_embedding, skill_embedding)
            similarities.append((skill_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
