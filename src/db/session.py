"""
Database Session Management

Connection pooling and session lifecycle.
"""

from contextlib import contextmanager
from typing import Generator
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base


# Default database URL
DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///letta_rl.db"
)


class DatabaseSession:
    """
    Database session manager with connection pooling.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for database connection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        database_url: str = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ):
        if hasattr(self, '_initialized'):
            return
        
        self.database_url = database_url or DEFAULT_DATABASE_URL
        
        # Configure engine based on database type
        if self.database_url.startswith("sqlite"):
            # SQLite doesn't support connection pooling the same way
            self.engine = create_engine(
                self.database_url,
                echo=echo,
                connect_args={"check_same_thread": False},
            )
        else:
            # PostgreSQL or other databases
            self.engine = create_engine(
                self.database_url,
                echo=echo,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
        
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
        
        self._initialized = True
    
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        """Drop all tables."""
        Base.metadata.drop_all(self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db.get_session() as session:
                session.query(...)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_raw_session(self) -> Session:
        """
        Get a raw session (caller must manage lifecycle).
        
        Use get_session() context manager when possible.
        """
        return self.SessionLocal()


# Global instance
_db: DatabaseSession = None


def init_db(database_url: str = None, **kwargs) -> DatabaseSession:
    """Initialize the global database instance."""
    global _db
    _db = DatabaseSession(database_url, **kwargs)
    _db.create_tables()
    return _db


def get_db() -> DatabaseSession:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = DatabaseSession()
        _db.create_tables()
    return _db


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Convenience function for getting a session."""
    db = get_db()
    with db.get_session() as session:
        yield session


# Dependency for FastAPI
def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    db = get_db()
    session = db.SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
