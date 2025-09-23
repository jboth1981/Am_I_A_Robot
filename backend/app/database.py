from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Database connection string from environment variable
# This automatically adapts to local vs production environments
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://robot_user:local_password@localhost:5432/am_i_a_robot_local")

# Create the database engine
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency function for FastAPI
def get_db():
    """FastAPI dependency that provides a database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()