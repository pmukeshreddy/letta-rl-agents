# 🧠 Letta RL Skill Selector

**RL-optimized skill selection for [Letta](https://github.com/letta-ai/letta) agents.**

> Letta agents learn *how* to create skills. This teaches them *when* to use them.

## The Problem

Letta's Skill Learning creates `.md` skill files from experience. But skill **selection** is pure LLM guesswork — no optimization, no feedback loop, no multi-agent transfer.

## What This Does

| Component | Description |
|-----------|-------------|
| **RL Skill Selector** | PPO policy that learns *which* skills to load for each task |
| **Quality Tracker** | Tracks skill effectiveness with success rates & rewards |
| **Semantic Matching** | Embedding-based task-skill similarity |
| **Feedback Loop** | Continuous improvement from task outcomes |
| **Multi-Agent Transfer** | Share proven skills across agents |

## How It Works

```
Task → RL Policy → Select Top-K Skills → Inject into Letta → Execute → Reward
           ↑                                                         ↓
           └──────────────────── Learn ────────────────────────────┘
```

## Quick Start

```bash
# Clone
git clone https://github.com/pmukeshreddy/letta-rl-agents.git
cd letta-rl-agents

# Install
pip install -e .

# Seed skills
python scripts/seed_skills.py

# Run (mock mode - no Letta API needed)
make api   # Terminal 1: API on :8000
make ui    # Terminal 2: UI on :7860
```

Open **http://localhost:7860**

## With Docker

```bash
cp .env.example .env
# Edit .env with your LETTA_API_KEY

docker-compose up
```

## Project Structure

```
letta-rl-agents/
├── src/
│   ├── selector/          # RL skill selection
│   │   ├── policy.py      # PPO actor-critic
│   │   ├── embeddings.py  # Task/skill embeddings
│   │   ├── trainer.py     # Training loop
│   │   └── buffer.py      # Experience replay
│   ├── tracker/           # Quality tracking
│   │   ├── quality.py     # Skill scoring
│   │   └── analytics.py   # Usage analytics
│   ├── agents/            # Letta integration
│   │   ├── client.py      # Letta API wrapper
│   │   ├── executor.py    # Task execution
│   │   └── skill_loader.py
│   ├── skills/            # Skill management
│   │   ├── repository.py  # CRUD operations
│   │   └── transfer.py    # Multi-agent transfer
│   ├── db/                # Database
│   │   ├── models.py      # SQLAlchemy models
│   │   └── session.py     # Connection management
│   └── api/               # REST API
│       ├── server.py      # FastAPI app
│       └── routes/        # Endpoints
├── skills/                # Skill library (.md files)
├── ui/dashboard.py        # Gradio interface
├── tests/                 # Test suite
└── scripts/               # Utilities
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tasks/execute` | Execute task with RL selection |
| POST | `/tasks/select` | Select skills without executing |
| GET | `/skills` | List all skills |
| POST | `/skills` | Create new skill |
| POST | `/training/run` | Run training update |
| GET | `/metrics/dashboard` | Get dashboard data |

## Usage Example

```python
from src.agents import SkillSelector, TaskExecutor, MockLettaClient
from src.db import init_db
from src.skills import SkillRepository

# Initialize
db = init_db()
selector = SkillSelector()
client = MockLettaClient()  # or LettaClient(api_key=...)
repo = SkillRepository(db)

executor = TaskExecutor(selector, client, repo)

# Execute task
result = executor.execute("Generate a PDF report from sales data")
print(f"Success: {result.success}")
print(f"Skills used: {result.skills_used}")

# Train on feedback
executor.train()
```

## Tech Stack

- **RL**: PPO with GAE (pure NumPy, CPU-only)
- **Embeddings**: sentence-transformers
- **Database**: PostgreSQL / SQLite
- **API**: FastAPI
- **UI**: Gradio

## Make Commands

```bash
make install    # Install package
make dev        # Install with dev deps
make api        # Run API server
make ui         # Run Gradio dashboard
make seed       # Load skills into DB
make embeddings # Generate embeddings
make test       # Run tests
make eval       # Run evaluation
make up         # Docker compose up
make down       # Docker compose down
```

## Comparison

| | Letta Today | + This Project |
|---|---|---|
| Skill Creation | LLM reflects → writes .md | Same |
| Skill Selection | LLM guessing | **RL policy learns** |
| Feedback Loop | None | **Reward → training** |
| Quality Tracking | None | **Success rates** |
| Multi-Agent | Skills stuck | **Shared repository** |


