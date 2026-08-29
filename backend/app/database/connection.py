"""
Database Connection & Session Manager
-------------------------------------
Configures the SQLite engine via SQLAlchemy with foreign key constraints enabled.
Provides session dependencies for FastAPI route handlers.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.app.config import settings

# SQLite connection string
DATABASE_URL = f"sqlite:///{settings.DATABASE_PATH.as_posix()}"

# Enable multi-threaded SQLite access for FastAPI
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)


# SQLite Foreign Key Enforcer
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI Dependency for obtaining a clean database session per request."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Creates all database tables defined in models.py."""
    Base.metadata.create_all(bind=engine)
