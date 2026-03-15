"""
database.py — SQLAlchemy Engine & Session Factory
==================================================
Reads DATABASE_URL from a .env file.  Defaults to a local MySQL DSN if absent.

Required .env keys:
  DATABASE_URL=mysql+pymysql://root:<password>@localhost:3306/aroma_twin_v2
  PERFUME_CSV_PATH=perfumes_final_1.csv

MySQL setup (run once):
  CREATE DATABASE aroma_twin_v2 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# ── Connection String ──────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@localhost:3306/aroma_twin_v2",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,   # reconnect on stale connections (long demo sessions)
    pool_recycle=1800,    # recycle every 30 min
    echo=False,           # set True to see raw SQL during debug
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# All ORM classes inherit from this Base; passed to create_all() in app.py
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that yields a scoped DB session and guarantees cleanup.
    Usage:  def route(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
