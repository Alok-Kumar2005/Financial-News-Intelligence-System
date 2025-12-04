from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from src.config import config
from src.models import Base
from loguru import logger

engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in config.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


@contextmanager
def get_db_session() -> Session:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # 1️⃣ Initialize database
    init_db()
    logger.info("DB initialized")

    # 2️⃣ Test contextmanager session
    with get_db_session() as session:
        logger.info("Context manager session opened successfully")
        print("Session OK:", session)

    # 3️⃣ Test FastAPI-like generator session
    db = next(get_db())
    print("FastAPI-style session OK:", db)
    db.close()

    logger.info("All DB tests completed successfully")