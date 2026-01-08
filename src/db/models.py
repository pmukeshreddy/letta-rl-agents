"""
Database Models

SQLAlchemy models for the RL skill selector.

Tables:
- skills: Skill definitions and embeddings
- skill_stats: Detailed quality metrics per skill
- tasks: Task execution records  
- experiences: RL training tuples (state, action, reward)
- policies: Versioned policy checkpoints
- training_runs: Training session metadata
"""

from datetime import datetime, timedelta
from typing import Optional
import json

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    LargeBinary,
    ForeignKey,
    Index,
    Table,
    func,
)
from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
    relationship,
    Session,
)
from sqlalchemy import JSON, ARRAY

Base = declarative_base()


# Association table for Experience <-> Skill
experience_skills = Table(
    "experience_skills",
    Base.metadata,
    Column("experience_id", Integer, ForeignKey("experiences.id"), primary_key=True),
    Column("skill_id", String(255), ForeignKey("skills.id"), primary_key=True),
)


class Skill(Base):
    """
    A skill that can be loaded into agent context.
    
    Skills are .md files containing domain knowledge,
    approaches, pitfalls, and examples for specific tasks.
    """
    __tablename__ = "skills"
    
    id = Column(String(255), primary_key=True)  # e.g., "pdf-generation"
    name = Column(String(255), nullable=False)
    description = Column(Text)
    content = Column(Text, nullable=False)  # Full .md content
    file_path = Column(String(512))
    
    # Embedding for similarity matching (serialized numpy array)
    embedding = Column(LargeBinary)
    embedding_model = Column(String(128), default="text-embedding-3-small")
    
    # Metadata
    category = Column(String(64))  # e.g., "file-ops", "api", "data"
    tags = Column(JSON, default=[])
    token_count = Column(Integer, default=0)  # For context budget planning
    
    # Denormalized stats (updated periodically for fast queries)
    total_uses = Column(Integer, default=0)
    successful_uses = Column(Integer, default=0)
    total_reward = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    stats = relationship("SkillStats", back_populates="skill", uselist=False, cascade="all, delete-orphan")
    task_outcomes = relationship("TaskOutcome", secondary="task_skills", back_populates="skills")
    
    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.successful_uses / self.total_uses
    
    @property
    def avg_reward(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.total_reward / self.total_uses
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "token_count": self.token_count,
            "success_rate": self.success_rate,
            "avg_reward": self.avg_reward,
            "total_uses": self.total_uses,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f"<Skill {self.id}: {self.name}>"


class SkillStats(Base):
    """
    Detailed statistics for skill quality tracking.
    
    Separate from Skill to allow detailed tracking without
    bloating the main skill queries.
    """
    __tablename__ = "skill_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    skill_id = Column(String(255), ForeignKey("skills.id"), unique=True, nullable=False)
    
    # Detailed usage counts
    total_uses = Column(Integer, default=0)
    successful_uses = Column(Integer, default=0)
    failed_uses = Column(Integer, default=0)
    
    # Reward tracking
    total_reward = Column(Float, default=0.0)
    min_reward = Column(Float, nullable=True)
    max_reward = Column(Float, nullable=True)
    reward_variance = Column(Float, default=0.0)
    
    # Performance by task type (JSON: {"coding": {"uses": 10, "success": 8}})
    task_type_stats = Column(JSON, default={})
    
    # Co-occurrence with other skills
    co_occurrence_stats = Column(JSON, default={})  # {"other_skill_id": {"count": 5, "success": 4}}
    
    # Time-based tracking
    last_used_at = Column(DateTime, nullable=True)
    last_success_at = Column(DateTime, nullable=True)
    last_failure_at = Column(DateTime, nullable=True)
    
    # Rolling window stats (last 7 days)
    recent_uses = Column(Integer, default=0)
    recent_successes = Column(Integer, default=0)
    recent_success_rate = Column(Float, default=0.0)
    trend = Column(String(16), default="stable")  # "improving", "declining", "stable"
    
    # Relationships
    skill = relationship("Skill", back_populates="stats")
    
    @property
    def success_rate(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.successful_uses / self.total_uses
    
    @property
    def avg_reward(self) -> float:
        if self.total_uses == 0:
            return 0.0
        return self.total_reward / self.total_uses
    
    def __repr__(self):
        return f"<SkillStats {self.skill_id}: {self.success_rate:.1%}>"


class TaskOutcome(Base):
    """
    A task execution record.
    
    Stores the task, selected skills, outcome, and all metadata
    needed for RL training and analysis.
    """
    __tablename__ = "task_outcomes"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Task details
    task_description = Column(Text, nullable=False)
    task_type = Column(String(64))  # Inferred category: "coding", "writing", etc.
    task_embedding = Column(LargeBinary)  # For similarity analysis
    
    # Skill selection
    skills_available = Column(JSON, default=[])  # All skills that were available
    selection_method = Column(String(32), default="rl")  # "rl", "random", "manual", "baseline"
    selection_confidence = Column(JSON, default={})  # {"skill_id": 0.85, ...}
    
    # Execution context
    agent_id = Column(String(255))
    model = Column(String(255))
    policy_version = Column(Integer)  # Which policy version made the selection
    
    # Outcome
    success = Column(Boolean, nullable=False)
    reward = Column(Float, default=0.0)
    reward_components = Column(JSON, default={})  # {"task_success": 1.0, "efficiency": 0.2}
    
    # Response data
    response = Column(Text)
    error_message = Column(Text, nullable=True)
    
    # Metrics
    tokens_used = Column(Integer, default=0)
    context_tokens = Column(Integer, default=0)  # Tokens used by skills
    latency_ms = Column(Integer, default=0)
    
    # User feedback (optional)
    user_feedback = Column(String(16), nullable=True)  # "positive", "negative"
    feedback_text = Column(Text, nullable=True)
    feedback_at = Column(DateTime, nullable=True)
    
    # Additional metadata
    task_metadata = Column(JSON, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    skills = relationship("Skill", secondary="task_skills", back_populates="task_outcomes")
    experience = relationship("Experience", back_populates="task", uselist=False)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "success": self.success,
            "reward": self.reward,
            "skills_used": [s.id for s in self.skills],
            "selection_method": self.selection_method,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f"<TaskOutcome {self.task_id[:8]}: success={self.success}>"


class TaskSkill(Base):
    """Association table for task outcomes and skills with metadata."""
    __tablename__ = "task_skills"
    
    task_outcome_id = Column(Integer, ForeignKey("task_outcomes.id"), primary_key=True)
    skill_id = Column(String(255), ForeignKey("skills.id"), primary_key=True)
    
    # Selection metadata
    confidence_score = Column(Float)  # Policy confidence for this skill
    selection_rank = Column(Integer)  # Order in which skill was selected (1 = highest)
    was_loaded = Column(Boolean, default=True)  # Was actually loaded (vs. skipped for budget)


class Experience(Base):
    """
    RL experience tuple for PPO training.
    
    Stores complete (state, action, reward) tuples with
    all information needed for policy gradient updates.
    """
    __tablename__ = "experiences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_outcome_id = Column(Integer, ForeignKey("task_outcomes.id"), unique=True)
    
    # State representation
    state_task_embedding = Column(LargeBinary)  # Task embedding
    state_available_skills = Column(JSON)  # List of available skill IDs
    state_skill_qualities = Column(JSON)  # {"skill_id": quality_score} at decision time
    state_context_budget = Column(Integer)  # Available tokens for skills
    
    # Action taken
    action_selected_skills = Column(JSON)  # Ordered list of selected skill IDs
    action_probabilities = Column(JSON)  # {"skill_id": selection_probability}
    action_log_probs = Column(JSON)  # {"skill_id": log_prob} for PPO
    
    # Reward
    reward = Column(Float, nullable=False)
    reward_components = Column(JSON, default={})
    
    # Value estimation (for advantage calculation)
    value_estimate = Column(Float, nullable=True)  # V(s) from critic
    advantage = Column(Float, nullable=True)  # A(s,a) = Q(s,a) - V(s)
    returns = Column(Float, nullable=True)  # Discounted returns
    
    # Training metadata
    policy_version = Column(Integer, nullable=False)  # Policy that generated this
    used_for_training = Column(Boolean, default=False)
    training_batch_id = Column(Integer, nullable=True)
    trained_at = Column(DateTime, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    task = relationship("TaskOutcome", back_populates="experience")
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_outcome_id": self.task_outcome_id,
            "action_selected_skills": self.action_selected_skills,
            "reward": self.reward,
            "advantage": self.advantage,
            "policy_version": self.policy_version,
            "used_for_training": self.used_for_training,
        }
    
    def __repr__(self):
        return f"<Experience {self.id}: r={self.reward:.2f}>"


class Policy(Base):
    """
    Versioned policy checkpoint.
    
    Stores trained policy weights for deployment, rollback,
    and A/B testing comparisons.
    """
    __tablename__ = "policies"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, nullable=False, unique=True)
    name = Column(String(255))  # Optional friendly name
    
    # Serialized weights (numpy arrays as bytes)
    weights = Column(LargeBinary, nullable=False)
    
    # Architecture config (for reconstruction)
    config = Column(JSON, nullable=False, default={
        "embedding_dim": 384,
        "hidden_dim": 256,
        "max_skills": 5,
    })
    
    # Training metadata
    trained_on_experiences = Column(Integer, default=0)
    training_epochs = Column(Integer, default=0)
    final_loss = Column(Float, nullable=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=True)
    
    # Evaluation metrics
    eval_success_rate = Column(Float, nullable=True)
    eval_avg_reward = Column(Float, nullable=True)
    eval_num_tasks = Column(Integer, default=0)
    
    # Comparison to baseline
    baseline_delta_success = Column(Float, nullable=True)  # vs random selection
    baseline_delta_reward = Column(Float, nullable=True)
    
    # Deployment status
    is_active = Column(Boolean, default=False)  # Currently deployed
    is_baseline = Column(Boolean, default=False)  # Baseline for comparison
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    activated_at = Column(DateTime, nullable=True)
    deactivated_at = Column(DateTime, nullable=True)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "trained_on_experiences": self.trained_on_experiences,
            "eval_success_rate": self.eval_success_rate,
            "eval_avg_reward": self.eval_avg_reward,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def __repr__(self):
        return f"<Policy v{self.version}: active={self.is_active}>"


class TrainingRun(Base):
    """
    Training run metadata.
    
    Tracks each training session for reproducibility,
    debugging, and hyperparameter analysis.
    """
    __tablename__ = "training_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Policy versions
    policy_version_start = Column(Integer, nullable=False)
    policy_version_end = Column(Integer, nullable=True)
    
    # Hyperparameters
    config = Column(JSON, default={})
    learning_rate = Column(Float, default=3e-4)
    batch_size = Column(Integer, default=64)
    epochs = Column(Integer, default=10)
    clip_epsilon = Column(Float, default=0.2)  # PPO clipping
    gamma = Column(Float, default=0.99)  # Discount factor
    gae_lambda = Column(Float, default=0.95)  # GAE parameter
    entropy_coef = Column(Float, default=0.01)  # Entropy bonus
    value_coef = Column(Float, default=0.5)  # Value loss coefficient
    
    # Training data
    num_experiences = Column(Integer, default=0)
    experience_id_start = Column(Integer, nullable=True)
    experience_id_end = Column(Integer, nullable=True)
    
    # Loss tracking
    loss_initial = Column(Float, nullable=True)
    loss_final = Column(Float, nullable=True)
    policy_loss_final = Column(Float, nullable=True)
    value_loss_final = Column(Float, nullable=True)
    entropy_final = Column(Float, nullable=True)
    
    # Performance metrics
    success_rate_before = Column(Float, nullable=True)
    success_rate_after = Column(Float, nullable=True)
    avg_reward_before = Column(Float, nullable=True)
    avg_reward_after = Column(Float, nullable=True)
    
    # Status
    status = Column(String(16), default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    policies = relationship("Policy", backref="training_run")
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "policy_version_start": self.policy_version_start,
            "policy_version_end": self.policy_version_end,
            "num_experiences": self.num_experiences,
            "loss_final": self.loss_final,
            "success_rate_before": self.success_rate_before,
            "success_rate_after": self.success_rate_after,
        }
    
    def __repr__(self):
        return f"<TrainingRun {self.id}: {self.status}>"


class TrainingMetrics(Base):
    """
    Per-step training metrics.
    
    Logged during training for visualization and debugging.
    """
    __tablename__ = "training_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), nullable=False)
    
    # Step info
    epoch = Column(Integer, nullable=False)
    batch = Column(Integer, nullable=False)
    global_step = Column(Integer, nullable=False, index=True)
    
    # Losses
    total_loss = Column(Float)
    policy_loss = Column(Float)
    value_loss = Column(Float)
    entropy = Column(Float)
    
    # Gradient stats
    grad_norm = Column(Float, nullable=True)
    
    # PPO-specific
    clip_fraction = Column(Float, nullable=True)  # Fraction of clipped ratios
    approx_kl = Column(Float, nullable=True)  # Approximate KL divergence
    
    # Performance snapshot
    avg_reward = Column(Float, nullable=True)
    avg_advantage = Column(Float, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)


# Indexes for performance
Index("idx_task_outcomes_created", TaskOutcome.created_at)
Index("idx_task_outcomes_success", TaskOutcome.success)
Index("idx_task_outcomes_policy", TaskOutcome.policy_version)
Index("idx_skills_category", Skill.category)
Index("idx_experiences_policy", Experience.policy_version)
Index("idx_experiences_training", Experience.used_for_training)
Index("idx_policies_active", Policy.is_active)


class Database:
    """
    Database connection manager.
    
    Handles connection pooling and session management.
    """
    
    def __init__(self, url: str = "sqlite:///letta_rl.db"):
        self.url = url
        self.engine = create_engine(
            url,
            pool_pre_ping=True,  # Check connection health
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
    
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(self.engine)
    
    def drop_tables(self):
        """Drop all tables (dangerous!)."""
        Base.metadata.drop_all(self.engine)
    
    def get_session(self) -> Session:
        """Get a new session."""
        return self.SessionLocal()
    
    def healthcheck(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception:
            return False


class SkillRepository:
    """Repository for skill CRUD operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create(self, skill: Skill) -> Skill:
        """Create a new skill."""
        with self.db.get_session() as session:
            # Also create empty stats
            skill.stats = SkillStats(skill_id=skill.id)
            session.add(skill)
            session.commit()
            session.refresh(skill)
            return skill
    
    def get(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID."""
        with self.db.get_session() as session:
            return session.query(Skill).filter(Skill.id == skill_id).first()
    
    def get_with_stats(self, skill_id: str) -> Optional[Skill]:
        """Get skill with stats eagerly loaded."""
        with self.db.get_session() as session:
            from sqlalchemy.orm import joinedload
            return (
                session.query(Skill)
                .options(joinedload(Skill.stats))
                .filter(Skill.id == skill_id)
                .first()
            )
    
    def list_all(self) -> list[Skill]:
        """List all skills."""
        with self.db.get_session() as session:
            return session.query(Skill).order_by(Skill.name).all()
    
    def list_by_category(self, category: str) -> list[Skill]:
        """List skills by category."""
        with self.db.get_session() as session:
            return (
                session.query(Skill)
                .filter(Skill.category == category)
                .order_by(Skill.name)
                .all()
            )
    
    def get_top_skills(
        self,
        limit: int = 10,
        min_uses: int = 3,
    ) -> list[Skill]:
        """Get top performing skills."""
        with self.db.get_session() as session:
            return (
                session.query(Skill)
                .filter(Skill.total_uses >= min_uses)
                .order_by(
                    (Skill.successful_uses.cast(Float) / Skill.total_uses).desc()
                )
                .limit(limit)
                .all()
            )
    
    def update_stats(
        self,
        skill_id: str,
        success: bool,
        reward: float,
        task_type: Optional[str] = None,
    ):
        """Update skill stats after task execution."""
        with self.db.get_session() as session:
            skill = session.query(Skill).filter(Skill.id == skill_id).first()
            if not skill:
                return
            
            # Update denormalized stats
            skill.total_uses += 1
            if success:
                skill.successful_uses += 1
            skill.total_reward += reward
            skill.updated_at = datetime.utcnow()
            
            # Update detailed stats
            if skill.stats:
                stats = skill.stats
                stats.total_uses += 1
                if success:
                    stats.successful_uses += 1
                else:
                    stats.failed_uses += 1
                stats.total_reward += reward
                stats.last_used_at = datetime.utcnow()
                if success:
                    stats.last_success_at = datetime.utcnow()
                else:
                    stats.last_failure_at = datetime.utcnow()
                
                # Update min/max reward
                if stats.min_reward is None or reward < stats.min_reward:
                    stats.min_reward = reward
                if stats.max_reward is None or reward > stats.max_reward:
                    stats.max_reward = reward
                
                # Update task type stats
                if task_type:
                    type_stats = stats.task_type_stats or {}
                    if task_type not in type_stats:
                        type_stats[task_type] = {"uses": 0, "success": 0}
                    type_stats[task_type]["uses"] += 1
                    if success:
                        type_stats[task_type]["success"] += 1
                    stats.task_type_stats = type_stats
            
            session.commit()
    
    def update_embedding(self, skill_id: str, embedding: bytes):
        """Update skill embedding."""
        with self.db.get_session() as session:
            skill = session.query(Skill).filter(Skill.id == skill_id).first()
            if skill:
                skill.embedding = embedding
                skill.updated_at = datetime.utcnow()
                session.commit()
    
    def delete(self, skill_id: str) -> bool:
        """Delete a skill."""
        with self.db.get_session() as session:
            skill = session.query(Skill).filter(Skill.id == skill_id).first()
            if skill:
                session.delete(skill)
                session.commit()
                return True
            return False


class TaskRepository:
    """Repository for task outcome operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create(
        self,
        task_id: str,
        task_description: str,
        success: bool,
        reward: float,
        skill_ids: list[str],
        **kwargs,
    ) -> TaskOutcome:
        """Create a new task outcome."""
        with self.db.get_session() as session:
            outcome = TaskOutcome(
                task_id=task_id,
                task_description=task_description,
                success=success,
                reward=reward,
                **kwargs,
            )
            
            # Link skills
            for i, skill_id in enumerate(skill_ids):
                skill = session.query(Skill).filter(Skill.id == skill_id).first()
                if skill:
                    outcome.skills.append(skill)
                    # Create task_skill entry with metadata
                    task_skill = session.query(TaskSkill).filter(
                        TaskSkill.task_outcome_id == outcome.id,
                        TaskSkill.skill_id == skill_id,
                    ).first()
                    if task_skill:
                        task_skill.selection_rank = i + 1
            
            session.add(outcome)
            session.commit()
            session.refresh(outcome)
            return outcome
    
    def get(self, task_id: str) -> Optional[TaskOutcome]:
        """Get task by ID."""
        with self.db.get_session() as session:
            return session.query(TaskOutcome).filter(TaskOutcome.task_id == task_id).first()
    
    def get_recent(self, limit: int = 100) -> list[TaskOutcome]:
        """Get recent task outcomes."""
        with self.db.get_session() as session:
            return (
                session.query(TaskOutcome)
                .order_by(TaskOutcome.created_at.desc())
                .limit(limit)
                .all()
            )
    
    def get_by_policy_version(self, version: int, limit: int = 100) -> list[TaskOutcome]:
        """Get tasks executed by a specific policy version."""
        with self.db.get_session() as session:
            return (
                session.query(TaskOutcome)
                .filter(TaskOutcome.policy_version == version)
                .order_by(TaskOutcome.created_at.desc())
                .limit(limit)
                .all()
            )
    
    def get_success_rate(self, days: int = 7) -> float:
        """Get success rate over recent days."""
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            total = (
                session.query(func.count(TaskOutcome.id))
                .filter(TaskOutcome.created_at >= cutoff)
                .scalar()
            )
            
            successful = (
                session.query(func.count(TaskOutcome.id))
                .filter(TaskOutcome.created_at >= cutoff)
                .filter(TaskOutcome.success == True)
                .scalar()
            )
            
            if total == 0:
                return 0.0
            return successful / total
    
    def get_avg_reward(self, days: int = 7) -> float:
        """Get average reward over recent days."""
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            result = (
                session.query(func.avg(TaskOutcome.reward))
                .filter(TaskOutcome.created_at >= cutoff)
                .scalar()
            )
            
            return result or 0.0


class ExperienceRepository:
    """Repository for RL experience operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create(self, experience: Experience) -> Experience:
        """Create a new experience."""
        with self.db.get_session() as session:
            session.add(experience)
            session.commit()
            session.refresh(experience)
            return experience
    
    def get_for_training(
        self,
        limit: int = 1000,
        policy_version: Optional[int] = None,
    ) -> list[Experience]:
        """Get untrained experiences for training."""
        with self.db.get_session() as session:
            query = (
                session.query(Experience)
                .filter(Experience.used_for_training == False)
            )
            if policy_version:
                query = query.filter(Experience.policy_version == policy_version)
            
            return query.order_by(Experience.created_at.asc()).limit(limit).all()
    
    def mark_as_trained(self, experience_ids: list[int], batch_id: int):
        """Mark experiences as used for training."""
        with self.db.get_session() as session:
            session.query(Experience).filter(
                Experience.id.in_(experience_ids)
            ).update({
                "used_for_training": True,
                "training_batch_id": batch_id,
                "trained_at": datetime.utcnow(),
            }, synchronize_session=False)
            session.commit()
    
    def get_stats(self) -> dict:
        """Get experience statistics."""
        with self.db.get_session() as session:
            total = session.query(func.count(Experience.id)).scalar()
            untrained = (
                session.query(func.count(Experience.id))
                .filter(Experience.used_for_training == False)
                .scalar()
            )
            avg_reward = session.query(func.avg(Experience.reward)).scalar()
            
            return {
                "total": total or 0,
                "untrained": untrained or 0,
                "trained": (total or 0) - (untrained or 0),
                "avg_reward": avg_reward or 0.0,
            }


class PolicyRepository:
    """Repository for policy operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create(self, policy: Policy) -> Policy:
        """Create a new policy checkpoint."""
        with self.db.get_session() as session:
            session.add(policy)
            session.commit()
            session.refresh(policy)
            return policy
    
    def get_active(self) -> Optional[Policy]:
        """Get the currently active policy."""
        with self.db.get_session() as session:
            return (
                session.query(Policy)
                .filter(Policy.is_active == True)
                .first()
            )
    
    def get_by_version(self, version: int) -> Optional[Policy]:
        """Get policy by version number."""
        with self.db.get_session() as session:
            return session.query(Policy).filter(Policy.version == version).first()
    
    def get_latest(self) -> Optional[Policy]:
        """Get the most recent policy."""
        with self.db.get_session() as session:
            return (
                session.query(Policy)
                .order_by(Policy.version.desc())
                .first()
            )
    
    def get_latest_version(self) -> int:
        """Get the latest version number."""
        with self.db.get_session() as session:
            result = session.query(func.max(Policy.version)).scalar()
            return result or 0
    
    def activate(self, version: int) -> bool:
        """Activate a policy version (deactivate others)."""
        with self.db.get_session() as session:
            # Deactivate all
            session.query(Policy).update({"is_active": False})
            
            # Activate specified version
            policy = session.query(Policy).filter(Policy.version == version).first()
            if policy:
                policy.is_active = True
                policy.activated_at = datetime.utcnow()
                session.commit()
                return True
            return False
    
    def list_all(self, limit: int = 20) -> list[Policy]:
        """List all policies."""
        with self.db.get_session() as session:
            return (
                session.query(Policy)
                .order_by(Policy.version.desc())
                .limit(limit)
                .all()
            )
