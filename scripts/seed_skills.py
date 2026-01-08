#!/usr/bin/env python3
"""
Seed Skills

Load skill .md files into the database.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import init_db
from src.skills.repository import SkillRepository
from src.agents.skill_loader import SkillLoader


def main():
    """Load skills from files into database."""
    # Initialize database
    db = init_db()
    repo = SkillRepository(db)
    loader = SkillLoader("skills")
    
    print("Loading skills from ./skills/")
    
    skills = loader.load_all()
    created = 0
    updated = 0
    
    for skill_data in skills:
        existing = repo.get(skill_data["id"])
        
        if existing:
            repo.update(
                skill_data["id"],
                name=skill_data["name"],
                content=skill_data["content"],
                description=skill_data["description"],
                category=skill_data["category"],
                token_count=skill_data["token_count"],
            )
            updated += 1
            print(f"  Updated: {skill_data['id']}")
        else:
            repo.create(
                id=skill_data["id"],
                name=skill_data["name"],
                content=skill_data["content"],
                description=skill_data["description"],
                category=skill_data["category"],
                tags=skill_data.get("tags", []),
                token_count=skill_data["token_count"],
                file_path=skill_data["file_path"],
            )
            created += 1
            print(f"  Created: {skill_data['id']}")
    
    print(f"\nDone! Created: {created}, Updated: {updated}")


if __name__ == "__main__":
    main()
