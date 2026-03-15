"""
models.py — ORM Table Definitions (CRM-centric) & Pydantic v2 API Schemas
==========================================================================
CRM TABLES
  users            : Phone number is the PK — one record per customer ever.
  scent_passports  : The live, evolving MAUT vector (one row per accord label).
  calibration_logs : Records the 1-10 ratings given to 5 anchor perfumes on first visit.
  interaction_logs : Records per-round ratings during the refinement loop (any visit).
  purchases        : The final purchase event — one row per transaction.

BUG FIX NOTES (v2.1)
  RefineRequest.round_number was Field(..., ge=1, le=3).

  ROOT CAUSE OF 422 on /refine
  ─────────────────────────────────────────────────────────────────────────────
  After calibration the server returns round_number = 0.
  renderRefinementActions(0, ...) then set:
      refBtn.onclick = () => submitRefinement(0 + 1)   // nextRound = 1
  Inside submitRefinement(1):
      round_number: nextRound - 1  →  round_number = 0
  Pydantic validation: ge=1 rejects 0  →  422 Unprocessable Entity.

  FIXES APPLIED
  • round_number: ge=0  (accepts 0 for the first post-calibration refinement)
  • le=3 removed        (no server-side round cap — infinite refinement)
  ─────────────────────────────────────────────────────────────────────────────
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict

from sqlalchemy import Column, String, Float, DateTime, Integer, ForeignKey
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, field_validator

from database import Base


# ══════════════════════════════════════════════════════════════════════════════
#  SQLALCHEMY ORM — PHYSICAL TABLES
# ══════════════════════════════════════════════════════════════════════════════

class User(Base):
    """
    One row per phone number.
    visit_count increments on every login call for a returning customer.
    """
    __tablename__ = "users"

    phone_number = Column(String(20), primary_key=True, nullable=False)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
    visit_count  = Column(Integer,  default=1,              nullable=False)

    scent_passports  = relationship("ScentPassport",  back_populates="user", cascade="all, delete-orphan")
    calibration_logs = relationship("CalibrationLog", back_populates="user", cascade="all, delete-orphan")
    interaction_logs = relationship("InteractionLog", back_populates="user", cascade="all, delete-orphan")
    purchases        = relationship("Purchase",       back_populates="user", cascade="all, delete-orphan")


class ScentPassport(Base):
    """
    Live MAUT vector stored as individual (accord_name, weight) rows.
    Mutated after every calibration and every Rocchio refinement round.
    """
    __tablename__ = "scent_passports"

    id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    phone_number = Column(String(20), ForeignKey("users.phone_number"), nullable=False)
    accord_name  = Column(String(100), nullable=False)
    weight       = Column(Float,       nullable=False)

    user = relationship("User", back_populates="scent_passports")


class CalibrationLog(Base):
    """Five rows per user (one per anchor perfume) written at first calibration."""
    __tablename__ = "calibration_logs"

    id                 = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    phone_number       = Column(String(20), ForeignKey("users.phone_number"), nullable=False)
    anchor_perfume_key = Column(String(200), nullable=False)
    rating             = Column(Float,       nullable=False)
    logged_at          = Column(DateTime,    default=datetime.utcnow)

    user = relationship("User", back_populates="calibration_logs")


class InteractionLog(Base):
    """
    One row per rated suggestion during refinement.
    round_number is unbounded — mirrors the infinite-refinement design.
    """
    __tablename__ = "interaction_logs"

    id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    phone_number = Column(String(20), ForeignKey("users.phone_number"), nullable=False)
    perfume_name = Column(String(200), nullable=False)
    rating       = Column(Float,      nullable=False)
    round_number = Column(Integer,    nullable=False)   # 0-based, unbounded
    logged_at    = Column(DateTime,   default=datetime.utcnow)

    user = relationship("User", back_populates="interaction_logs")


class Purchase(Base):
    """Final purchase event — one row per transaction."""
    __tablename__ = "purchases"

    id           = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    phone_number = Column(String(20), ForeignKey("users.phone_number"), nullable=False)
    perfume_name = Column(String(200), nullable=False)
    brand        = Column(String(200), nullable=False)
    purchased_at = Column(DateTime,   default=datetime.utcnow)

    user = relationship("User", back_populates="purchases")


# ══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC v2 SCHEMAS — REQUEST / RESPONSE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    phone_number: str = Field(
        ..., min_length=7, max_length=20,
        description="Customer's mobile number (digits only)",
    )

    @field_validator("phone_number")
    @classmethod
    def strip_non_digits(cls, v: str) -> str:
        cleaned = "".join(c for c in v if c.isdigit())
        if len(cleaned) < 7:
            raise ValueError("Phone number must have at least 7 digits")
        return cleaned


class AnchorRating(BaseModel):
    perfume_key: str   = Field(..., description="CSV 'Perfume' key e.g. 'light-blue'")
    rating:      float = Field(..., ge=1.0, le=10.0)


class CalibrateRequest(BaseModel):
    phone_number: str
    ratings: List[AnchorRating] = Field(..., min_length=5, max_length=5)


class RoundRating(BaseModel):
    """One perfume name + the rating given during a refinement round."""
    perfume_name: str
    rating:       float = Field(..., ge=1.0, le=10.0)


class RefineRequest(BaseModel):
    """
    POST /refine payload.

    round_number  ge=0 (FIXED from ge=1 — see module docstring).
                  No upper bound — infinite refinement supported.
    ratings       The 1-to-3 perfumes the user rated this round.
    gender_filter Optional. Passed through to the recommendation engine.
    """
    phone_number:  str
    round_number:  int = Field(
        ...,
        ge=0,                       # ← KEY FIX: was ge=1, caused 422 on round 0
        description="Round being completed (0-based, no upper limit)",
    )
    ratings:       List[RoundRating] = Field(..., min_length=1)
    gender_filter: Optional[str]     = Field(None, description="men | women | unisex | null")


class CheckoutRequest(BaseModel):
    phone_number: str
    perfume_name: str
    brand:        str


class ResetRequest(BaseModel):
    phone_number: str


# ── Response Schemas ──────────────────────────────────────────────────────────

class LoginResponse(BaseModel):
    phone_number: str
    is_new_user:  bool
    visit_count:  int
    message:      str


class PerfumeCard(BaseModel):
    rank:               int
    name:               str
    brand:              str
    country:            str
    gender:             str
    top_notes:          str
    heart_notes:        str
    base_notes:         str
    fragrantica_url:    str
    cosine_score:       float
    jaccard_score:      float
    explainability_tag: str
    matched_accords:    List[str]
    is_wildcard:        bool = False


class RecommendationResponse(BaseModel):
    """
    Returned by /calibrate, /refine, and /revisit.

    has_more_rounds is ALWAYS True — no server-side round cap.
    The refinement loop continues until the customer clicks Purchase.
    round_number is the zero-based index of the round just completed.
    """
    phone_number:    str
    recommendations: List[PerfumeCard]
    radar_data:      Dict[str, float]
    round_number:    int   # 0 = first recs (post-calibration or post-revisit)
    has_more_rounds: bool  # always True; kept for API forward-compatibility


class SimpleResponse(BaseModel):
    success: bool
    message: str


class AnchorPerfume(BaseModel):
    key:          str
    display_name: str
    brand:        str
    family_label: str
    family_key:   str
    top_notes:    str
    heart_notes:  str
    base_notes:   str
    accent_color: str
    icon:         str
