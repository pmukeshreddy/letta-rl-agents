# Database Operations

## Connection Management
- Use connection pooling (SQLAlchemy, asyncpg)
- Always use context managers
- Set connection timeouts
- Handle connection drops gracefully

## SQLAlchemy (Recommended)
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(
    "postgresql://user:pass@host/db",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
)
Session = sessionmaker(bind=engine)

# Always use sessions with context
with Session() as session:
    result = session.execute(query)
    session.commit()
```

## Query Patterns
- SELECT: Always specify columns, avoid `SELECT *`
- INSERT: Use `ON CONFLICT` for upserts
- UPDATE: Always include WHERE clause
- DELETE: Double-check WHERE clause before executing

## Common Pitfalls
- SQL injection (always use parameterized queries)
- N+1 queries (use eager loading)
- Missing indexes on frequently queried columns
- Not closing connections/sessions
- Forgetting to commit transactions

## Migration Best Practices
- Use Alembic for schema migrations
- Always have rollback scripts
- Test migrations on staging first
- Backup before major migrations

## Parameterized Queries
```python
# NEVER do this:
f"SELECT * FROM users WHERE id = {user_id}"  # SQL injection!

# Always do this:
session.execute(
    "SELECT * FROM users WHERE id = :id",
    {"id": user_id}
)
```

## Verification
- Test queries return expected results
- Check execution time for performance
- Verify indexes are being used (EXPLAIN)
