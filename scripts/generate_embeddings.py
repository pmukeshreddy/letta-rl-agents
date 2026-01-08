#!/usr/bin/env python3
"""
Generate Embeddings

Pre-compute embeddings for all skills.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import init_db
from src.skills.repository import SkillRepository
from src.selector.embeddings import EmbeddingModel, SkillEmbedder


def main():
    """Generate embeddings for all skills."""
    # Initialize
    db = init_db()
    repo = SkillRepository(db)
    
    print("Initializing embedding model...")
    model = EmbeddingModel(model_name="all-MiniLM-L6-v2")
    embedder = SkillEmbedder(model)
    
    # Get all skills
    skills = repo.list_all()
    print(f"Found {len(skills)} skills")
    
    for skill in skills:
        print(f"  Embedding: {skill.id}...", end=" ")
        
        embedding = embedder.embed_skill(skill.name, skill.content)
        repo.update_embedding(skill.id, embedding)
        
        print("done")
    
    print(f"\nGenerated embeddings for {len(skills)} skills")


if __name__ == "__main__":
    main()
