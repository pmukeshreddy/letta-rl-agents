#!/usr/bin/env python3
"""
Run Evaluation

Evaluate the RL skill selector against baselines.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import init_db
from src.skills.repository import SkillRepository
from src.agents.executor import SkillSelector, TaskExecutor
from src.agents.client import MockLettaClient


# Test tasks for evaluation
EVAL_TASKS = [
    "Generate a PDF report from sales data",
    "Parse JSON from an API response",
    "Write unit tests for the user service",
    "Debug the authentication middleware",
    "Scrape product prices from a website",
    "Process a CSV file and compute statistics",
    "Create a database migration script",
    "Handle errors in the payment flow",
    "Set up git hooks for the project",
    "Clean and normalize user input data",
]


def evaluate_selector(executor: TaskExecutor, tasks: list[str], label: str) -> dict:
    """Run evaluation on a set of tasks."""
    results = []
    
    for task in tasks:
        result = executor.execute(task)
        results.append({
            "task": task,
            "success": result.success,
            "reward": result.reward,
            "skills_used": result.skills_used,
            "latency_ms": result.latency_ms,
        })
    
    success_count = sum(1 for r in results if r["success"])
    avg_reward = sum(r["reward"] for r in results) / len(results)
    
    return {
        "label": label,
        "total_tasks": len(tasks),
        "successes": success_count,
        "success_rate": success_count / len(tasks),
        "avg_reward": avg_reward,
        "results": results,
    }


def main():
    """Run full evaluation."""
    print("=" * 60)
    print("Letta RL Skill Selector - Evaluation")
    print("=" * 60)
    
    # Initialize
    db = init_db()
    repo = SkillRepository(db)
    
    # Load skills
    skills = repo.list_all()
    print(f"\nLoaded {len(skills)} skills")
    
    # Create selector
    selector = SkillSelector()
    from src.agents.executor import Skill
    for s in skills:
        selector.register_skill(Skill(
            id=s.id,
            name=s.name,
            description=s.description or "",
            content=s.content,
            token_count=s.token_count,
            success_rate=s.success_rate,
        ))
    
    # Mock client with different success rates
    mock_client = MockLettaClient(success_rate=0.6)
    executor = TaskExecutor(selector, mock_client, repo)
    
    print(f"\nRunning evaluation on {len(EVAL_TASKS)} tasks...")
    print("-" * 40)
    
    # Evaluate RL selector
    rl_results = evaluate_selector(executor, EVAL_TASKS, "RL Selector")
    
    print(f"\n📊 Results: {rl_results['label']}")
    print(f"   Success Rate: {rl_results['success_rate']:.1%}")
    print(f"   Avg Reward: {rl_results['avg_reward']:.3f}")
    
    # Save results
    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_skills": len(skills),
        "evaluations": [rl_results],
    }
    
    output_path = Path("eval_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print per-task results
    print("\n" + "=" * 60)
    print("Per-Task Results")
    print("=" * 60)
    
    for r in rl_results["results"]:
        status = "✓" if r["success"] else "✗"
        skills = ", ".join(r["skills_used"][:3]) if r["skills_used"] else "none"
        print(f"{status} {r['task'][:40]:40s} | {skills}")


if __name__ == "__main__":
    main()
