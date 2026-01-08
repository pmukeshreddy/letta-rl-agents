#!/usr/bin/env python3
"""
Letta RL Agents - Quick Start Script

Run this to test the full system.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()


def test_letta_connection():
    """Test Letta API connection."""
    print("\n" + "="*60)
    print("1. TESTING LETTA API CONNECTION")
    print("="*60)
    
    api_key = os.getenv("LETTA_API_KEY")
    if not api_key:
        print("❌ LETTA_API_KEY not found in .env")
        return False
    
    print(f"✓ API key found: {api_key[:20]}...")
    
    try:
        from letta_client import Letta
        client = Letta(api_key=api_key)
        
        # Try to list agents
        agents = client.agents.list()
        print(f"✓ Connected to Letta API")
        agents_list = list(agents); print(f"  Existing agents: {len(agents_list)}")
        return True
        
    except ImportError:
        print("⚠️ letta package not installed")
        print("  Run: pip install letta")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def setup_database():
    """Initialize database."""
    print("\n" + "="*60)
    print("2. SETTING UP DATABASE")
    print("="*60)
    
    try:
        from src.db.session import init_db
        db = init_db()
        print("✓ Database initialized")
        return db
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return None


def load_skills(db):
    """Load skills into database."""
    print("\n" + "="*60)
    print("3. LOADING SKILLS")
    print("="*60)
    
    try:
        from src.skills.repository import SkillRepository
        from src.agents.skill_loader import SkillLoader
        
        repo = SkillRepository(db)
        loader = SkillLoader("skills")
        
        skills = loader.load_all()
        print(f"  Found {len(skills)} skill files")
        
        created = 0
        for skill_data in skills:
            existing = repo.get(skill_data["id"])
            if not existing:
                repo.create(
                    id=skill_data["id"],
                    name=skill_data["name"],
                    content=skill_data["content"],
                    description=skill_data["description"],
                    category=skill_data["category"],
                    tags=skill_data.get("tags", []),
                    token_count=skill_data["token_count"],
                )
                created += 1
                print(f"  + {skill_data['id']}")
        
        print(f"✓ Loaded {created} new skills")
        return repo
        
    except Exception as e:
        print(f"❌ Skill loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_selector():
    """Test the RL skill selector."""
    print("\n" + "="*60)
    print("4. TESTING RL SKILL SELECTOR")
    print("="*60)
    
    try:
        from src.agents.executor import SkillSelector, Skill
        import numpy as np
        
        selector = SkillSelector()
        
        # Register some test skills
        test_skills = [
            Skill(id="pdf-gen", name="PDF Generation", description="Create PDFs", content="...", token_count=100),
            Skill(id="api-int", name="API Integration", description="Call APIs", content="...", token_count=150),
            Skill(id="data-proc", name="Data Processing", description="Process data", content="...", token_count=120),
        ]
        
        for skill in test_skills:
            selector.register_skill(skill)
        
        print(f"✓ Registered {len(test_skills)} skills")
        
        # Test selection
        task = "Generate a PDF report from API data"
        selected, probs, _ = selector.select(task, temperature=1.0)
        
        print(f"✓ Task: '{task}'")
        print(f"  Selected: {selected}")
        print(f"  Probabilities: {probs}")
        
        return selector
        
    except Exception as e:
        print(f"❌ Selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_mock_evaluation():
    """Run evaluation with mock client."""
    print("\n" + "="*60)
    print("5. RUNNING MOCK EVALUATION")
    print("="*60)
    
    try:
        from src.agents.executor import SkillSelector, TaskExecutor, Skill
        from src.agents.client import MockLettaClient
        from src.db.session import init_db
        from src.skills.repository import SkillRepository
        
        # Setup
        db = init_db()
        repo = SkillRepository(db)
        selector = SkillSelector()
        mock_client = MockLettaClient(success_rate=0.6)
        
        # Load skills from DB
        skills = repo.list_all()
        for s in skills:
            selector.register_skill(Skill(
                id=s.id,
                name=s.name,
                description=s.description or "",
                content=s.content,
                token_count=s.token_count,
                success_rate=s.success_rate,
            ))
        
        print(f"  Loaded {len(skills)} skills")
        
        # Test tasks
        tasks = [
            "Generate a PDF report",
            "Parse JSON from API",
            "Write unit tests",
            "Debug authentication code",
            "Scrape website data",
        ]
        
        results = []
        for task in tasks:
            # Select skills
            selected, probs, _ = selector.select(task)
            
            # Execute with mock
            result = mock_client.execute_with_skills(
                task=task,
                skills=[{"id": sid, "name": sid, "content": "..."} for sid in selected],
            )
            
            status = "✓" if result["success"] else "✗"
            print(f"  {status} {task[:40]:40s} | {', '.join(selected[:2])}")
            
            results.append(result["success"])
        
        success_rate = sum(results) / len(results)
        print(f"\n✓ Mock evaluation complete")
        print(f"  Success rate: {success_rate:.1%}")
        
        return success_rate
        
    except Exception as e:
        print(f"❌ Mock evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_letta_evaluation():
    """Run evaluation with real Letta API."""
    print("\n" + "="*60)
    print("6. RUNNING LETTA API EVALUATION")
    print("="*60)
    
    api_key = os.getenv("LETTA_API_KEY")
    if not api_key:
        print("⚠️ Skipping - no API key")
        return None
    
    try:
        from src.agents.client import LettaClient
        from src.agents.executor import SkillSelector, Skill
        from src.db.session import init_db
        from src.skills.repository import SkillRepository
        
        # Setup
        db = init_db()
        repo = SkillRepository(db)
        selector = SkillSelector()
        letta = LettaClient(api_key=api_key)
        
        # Load skills
        skills = repo.list_all()
        for s in skills:
            selector.register_skill(Skill(
                id=s.id,
                name=s.name,
                description=s.description or "",
                content=s.content,
                token_count=s.token_count,
            ))
        
        print(f"  Loaded {len(skills)} skills")
        
        # Simple test
        tasks = [
            "What is 2 + 2?",
            "List 3 programming languages",
        ]
        
        results = []
        for task in tasks:
            selected, _, _ = selector.select(task)
            
            skill_contents = []
            for sid in selected[:2]:
                s = repo.get(sid)
                if s:
                    skill_contents.append({"id": s.id, "name": s.name, "content": s.content[:500]})
            
            result = letta.execute_with_skills(task, skill_contents)
            
            status = "✓" if result["success"] else "✗"
            response_preview = result.get("response", "")[:50]
            print(f"  {status} {task[:30]:30s} | {response_preview}...")
            
            results.append(result["success"])
        
        success_rate = sum(results) / len(results) if results else 0
        print(f"\n✓ Letta evaluation complete")
        print(f"  Success rate: {success_rate:.1%}")
        
        return success_rate
        
    except Exception as e:
        print(f"❌ Letta evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(results):
    """Print final summary."""
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"""
┌─────────────────────────────────────────┐
│  Letta RL Skill Selector - Results      │
├─────────────────────────────────────────┤
│  Database:     {"✓ Ready" if results.get("db") else "✗ Failed":24s} │
│  Skills:       {str(results.get("skills_count", 0)) + " loaded":24s} │
│  Selector:     {"✓ Working" if results.get("selector") else "✗ Failed":24s} │
│  Mock Eval:    {f"{results.get('mock_rate', 0):.1%} success" if results.get("mock_rate") else "✗ Failed":24s} │
│  Letta Eval:   {f"{results.get('letta_rate', 0):.1%} success" if results.get("letta_rate") else "⚠️ Skipped":24s} │
└─────────────────────────────────────────┘

Next steps:
  1. Run API:  uvicorn src.api.server:app --reload
  2. Run UI:   python ui/dashboard.py
  3. Open:     http://localhost:7860
""")


def main():
    """Run all tests."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║         LETTA RL SKILL SELECTOR - QUICK START             ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    results = {}
    
    # 1. Test Letta connection
    results["letta_connected"] = test_letta_connection()
    
    # 2. Setup database
    db = setup_database()
    results["db"] = db is not None
    
    # 3. Load skills
    if db:
        repo = load_skills(db)
        results["skills_count"] = len(repo.list_all()) if repo else 0
    
    # 4. Test selector
    selector = test_selector()
    results["selector"] = selector is not None
    
    # 5. Mock evaluation
    results["mock_rate"] = run_mock_evaluation()
    
    # 6. Letta evaluation (if connected)
    if results.get("letta_connected"):
        results["letta_rate"] = run_letta_evaluation()
    
    # Summary
    print_summary(results)


if __name__ == "__main__":
    main()
