"""
app.py — AromaTwin v2.1 FastAPI Application
============================================
Changes from v2.0:
  • Removed MAX_ROUNDS = 3 constant.
  • /refine  now always returns has_more_rounds=True  (infinite loop).
  • /revisit now always returns has_more_rounds=True  (was False).
    Returning customers can also refine indefinitely before purchasing.

Run locally:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from typing import Optional, List

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import engine, get_db, Base
from models import (
    User, ScentPassport, CalibrationLog, InteractionLog, Purchase, NotePreference,
    LoginRequest, LoginResponse,
    CalibrateRequest, RefineRequest, CheckoutRequest, ResetRequest,
    SaveNotesRequest, LoadNotesResponse,
    RecommendationResponse, PerfumeCard, SimpleResponse, AnchorPerfume,
)
from ml_engine import AromaTwinEngine

load_dotenv()

CSV_PATH: str = os.getenv("PERFUME_CSV_PATH", "perfumes_final_1.csv")

# Module-level engine reference — populated at startup
_engine: Optional[AromaTwinEngine] = None
_anchor_cache: Optional[List[dict]] = None


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _anchor_cache
    print("[STARTUP] Creating MySQL tables …")
    Base.metadata.create_all(bind=engine)
    print("[STARTUP] Loading ML engine …")
    _engine       = AromaTwinEngine(csv_path=CSV_PATH)
    _anchor_cache = _engine.get_anchor_data()
    print(f"[STARTUP] Done — {len(_anchor_cache)} anchors loaded.")
    yield
    print("[SHUTDOWN] Graceful exit.")


app = FastAPI(
    title="AromaTwin v2.1 — Omnichannel CRM Engine",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _guard_engine():
    if _engine is None:
        raise HTTPException(503, "ML engine still initialising. Retry in a moment.")


def _load_maut(db: Session, phone: str) -> dict:
    """Reconstruct the MAUT dict from scent_passport rows."""
    rows = db.query(ScentPassport).filter(
        ScentPassport.phone_number == phone
    ).all()
    return {r.accord_name: r.weight for r in rows}


def _save_maut(db: Session, phone: str, maut: dict):
    """Atomically replace all scent_passport rows for this user."""
    db.query(ScentPassport).filter(
        ScentPassport.phone_number == phone
    ).delete()
    for accord, weight in maut.items():
        db.add(ScentPassport(phone_number=phone, accord_name=accord, weight=weight))
    db.commit()


def _build_card(raw: dict) -> PerfumeCard:
    return PerfumeCard(**raw)


def _already_shown(db: Session, phone: str) -> List[str]:
    """All perfume names logged in interaction_logs for this user."""
    rows = db.query(InteractionLog.perfume_name).filter(
        InteractionLog.phone_number == phone
    ).all()
    return [r[0] for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", summary="Health Check")
def health():
    return {"status": "online", "version": "2.1.0"}


@app.get(
    "/anchors",
    response_model=List[AnchorPerfume],
    summary="Calibration anchor perfume metadata",
)
def get_anchors():
    """
    Returns the 5 calibration anchor scents.
    Called once on page load to populate the calibration carousel.
    Response is cached at startup — zero DB / ML work per call.
    """
    if not _anchor_cache:
        raise HTTPException(503, "Anchors not yet loaded.")
    return _anchor_cache


@app.post("/login", response_model=LoginResponse, summary="CRM login by phone number")
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """
    Checks whether the phone number exists.

    New user      → creates a row with visit_count=1.   Frontend → State 2 (Calibration).
    Returning     → increments visit_count.              Frontend → State 5 (Revisit).
    """
    user = db.query(User).filter(User.phone_number == payload.phone_number).first()

    if user is None:
        user = User(phone_number=payload.phone_number, visit_count=1)
        db.add(user)
        db.commit()
        db.refresh(user)
        return LoginResponse(
            phone_number=user.phone_number,
            is_new_user=True,
            visit_count=1,
            message="Welcome. Let us build your Scent Passport.",
        )
    else:
        user.visit_count += 1
        db.commit()
        db.refresh(user)
        return LoginResponse(
            phone_number=user.phone_number,
            is_new_user=False,
            visit_count=user.visit_count,
            message=f"Welcome back. Visit #{user.visit_count}.",
        )


@app.post(
    "/calibrate",
    response_model=RecommendationResponse,
    summary="Ingest anchor ratings and build initial Scent Passport",
)
def calibrate(payload: CalibrateRequest, db: Session = Depends(get_db)):
    """
    Called after the user rates all 5 anchor perfumes (State 2 → State 3).

    1. Persist 5 calibration ratings to calibration_logs.
    2. Build initial MAUT vector via calibrate_from_anchors().
    3. Persist MAUT to scent_passports.
    4. Run Two-Stage recommendation to produce first 3 suggestions.
    5. Return recommendations + radar_data.
       round_number = 0  (first recs, no refinement yet)
       has_more_rounds = True  (user can refine immediately)
    """
    _guard_engine()

    user = db.query(User).filter(User.phone_number == payload.phone_number).first()
    if not user:
        raise HTTPException(404, "User not found. Please call /login first.")

    # Overwrite any prior calibration for this session
    db.query(CalibrationLog).filter(
        CalibrationLog.phone_number == payload.phone_number
    ).delete()

    anchor_ratings_dict: dict = {}
    for r in payload.ratings:
        db.add(CalibrationLog(
            phone_number=payload.phone_number,
            anchor_perfume_key=r.perfume_key,
            rating=r.rating,
        ))
        anchor_ratings_dict[r.perfume_key] = r.rating

    db.commit()

    maut = _engine.calibrate_from_anchors(anchor_ratings_dict)
    _save_maut(db, payload.phone_number, maut)

    raw_recs = _engine.recommend(
        maut=maut,
        excluded_names=[],
        gender_filter=None,
        top_n=3,
    )

    return RecommendationResponse(
        phone_number=payload.phone_number,
        recommendations=[_build_card(r) for r in raw_recs],
        radar_data=_engine.get_radar_data(maut),
        round_number=0,
        has_more_rounds=True,   # always True — no cap
    )


@app.post(
    "/refine",
    response_model=RecommendationResponse,
    summary="Log round ratings, apply Rocchio update, return improved recommendations",
)
def refine(payload: RefineRequest, db: Session = Depends(get_db)):
    """
    Called each time the user clicks 'Refine Further' (unbounded rounds).

    FIX v2.1 — round_number is now 0-based and unbounded.
    The first call from the frontend sends round_number=0 (completing round 0).
    Backend stores it, increments to next_round=1, and always returns
    has_more_rounds=True so the loop never ends server-side.

    1. Persist interaction ratings with the current round_number.
    2. Load MAUT from scent_passports.
    3. Apply Rocchio: V_new = V_old + α × (rating/10) × V_perfume_accords.
    4. Persist updated MAUT.
    5. Recommend, excluding all previously logged perfumes.
    6. Return next batch with has_more_rounds=True always.
    """
    _guard_engine()

    user = db.query(User).filter(User.phone_number == payload.phone_number).first()
    if not user:
        raise HTTPException(404, "User not found.")

    # Persist this round's ratings
    for r in payload.ratings:
        db.add(InteractionLog(
            phone_number=payload.phone_number,
            perfume_name=r.perfume_name,
            rating=r.rating,
            round_number=payload.round_number,
        ))
    db.commit()

    # Load + update MAUT via Rocchio
    current_maut = _load_maut(db, payload.phone_number)
    if not current_maut:
        raise HTTPException(400, "No Scent Passport found. Please calibrate first.")

    rated_pairs  = [(r.perfume_name, r.rating) for r in payload.ratings]
    updated_maut = _engine.rocchio_update(current_maut, rated_pairs)
    _save_maut(db, payload.phone_number, updated_maut)

    # Exclude everything the user has seen so far
    excluded = _already_shown(db, payload.phone_number)
    raw_recs = _engine.recommend(
        maut=updated_maut,
        excluded_names=excluded,
        gender_filter=payload.gender_filter,
        top_n=3,
    )

    next_round = payload.round_number + 1   # purely informational for the frontend

    return RecommendationResponse(
        phone_number=payload.phone_number,
        recommendations=[_build_card(r) for r in raw_recs],
        radar_data=_engine.get_radar_data(updated_maut),
        round_number=next_round,
        has_more_rounds=True,   # ← FIXED: was `next_round < MAX_ROUNDS`. Always True now.
    )


@app.post(
    "/revisit",
    response_model=RecommendationResponse,
    summary="Returning customer — 2 safe bets + 1 wildcard, then infinite refinement",
)
def revisit(phone_number: str, db: Session = Depends(get_db)):
    """
    Called immediately after a returning user logs in (is_new_user=False).
    Skips calibration — reads the saved Scent Passport.

    Slot 1 & 2 — highest Cosine + Jaccard matches (safe bets).
    Slot 3     — Wildcard: high base-note overlap, different top-note family.

    FIX v2.1 — has_more_rounds is now True (was False).
    The frontend will show rating inputs and a 'Refine Further' button on
    the revisit cards, enabling the same infinite loop as State 3.
    """
    _guard_engine()

    user = db.query(User).filter(User.phone_number == phone_number).first()
    if not user:
        raise HTTPException(404, "User not found.")

    maut = _load_maut(db, phone_number)
    if not maut:
        raise HTTPException(
            400,
            "No Scent Passport saved. Please calibrate first.",
        )

    purchased = [
        p.perfume_name
        for p in db.query(Purchase).filter(Purchase.phone_number == phone_number).all()
    ]

    # Safe bets: Top 2 via Two-Stage
    safe_recs = _engine.recommend(maut=maut, excluded_names=purchased, top_n=2)

    # Wildcard: Slot 3
    wildcard = _engine.wildcard_recommend(
        maut=maut,
        excluded_names=purchased + [r["name"] for r in safe_recs],
    )

    all_recs = safe_recs
    if wildcard:
        wildcard["rank"] = 3
        all_recs.append(wildcard)
    else:
        fallback = _engine.recommend(
            maut=maut,
            excluded_names=purchased + [r["name"] for r in safe_recs],
            top_n=1,
        )
        all_recs.extend(fallback)

    return RecommendationResponse(
        phone_number=phone_number,
        recommendations=[_build_card(r) for r in all_recs],
        radar_data=_engine.get_radar_data(maut),
        round_number=0,
        has_more_rounds=True,   # ← FIXED: was False. Returning customers can also refine.
    )


@app.post(
    "/notes/save",
    response_model=SimpleResponse,
    summary="Persist the full note-preference map for a user",
)
def save_notes(payload: SaveNotesRequest, db: Session = Depends(get_db)):
    """
    Atomically replaces all note_preferences rows for this user.
    Called by the frontend after every rating batch (calibration + each refine round).
    """
    user = db.query(User).filter(User.phone_number == payload.phone_number).first()
    if not user:
        raise HTTPException(404, "User not found.")

    db.query(NotePreference).filter(
        NotePreference.phone_number == payload.phone_number
    ).delete()

    for note_name, score in payload.preferences.items():
        db.add(NotePreference(
            phone_number=payload.phone_number,
            note_name=note_name,
            score=score,
        ))
    db.commit()

    return SimpleResponse(success=True, message="Note preferences saved.")


@app.get(
    "/notes/load",
    response_model=LoadNotesResponse,
    summary="Retrieve the stored note-preference map for a user",
)
def load_notes(phone_number: str, db: Session = Depends(get_db)):
    """
    Returns {note_name: score} for all saved note preferences.
    Returns an empty dict if no data exists yet (new user / after reset).
    """
    rows = db.query(NotePreference).filter(
        NotePreference.phone_number == phone_number
    ).all()
    return LoadNotesResponse(
        phone_number=phone_number,
        preferences={r.note_name: r.score for r in rows},
    )


@app.post(
    "/checkout",
    response_model=SimpleResponse,
    summary="Log the purchase",
)
def checkout(payload: CheckoutRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.phone_number == payload.phone_number).first()
    if not user:
        raise HTTPException(404, "User not found.")

    db.add(Purchase(
        phone_number=payload.phone_number,
        perfume_name=payload.perfume_name,
        brand=payload.brand,
    ))
    db.commit()

    return SimpleResponse(
        success=True,
        message=f"Purchase logged: {payload.brand} '{payload.perfume_name}'.",
    )


@app.post(
    "/reset",
    response_model=SimpleResponse,
    summary="Reset scent profile — wipes MAUT, calibration, and interaction logs",
)
def reset(payload: ResetRequest, db: Session = Depends(get_db)):
    """
    Wipes: scent_passports + calibration_logs + interaction_logs.
    Preserves: the users row (phone stays) and purchases history.
    After reset the frontend sends the user back to State 2 (Calibration).
    """
    user = db.query(User).filter(User.phone_number == payload.phone_number).first()
    if not user:
        raise HTTPException(404, "User not found.")

    db.query(ScentPassport).filter(ScentPassport.phone_number == payload.phone_number).delete()
    db.query(CalibrationLog).filter(CalibrationLog.phone_number == payload.phone_number).delete()
    db.query(InteractionLog).filter(InteractionLog.phone_number == payload.phone_number).delete()
    db.query(NotePreference).filter(NotePreference.phone_number == payload.phone_number).delete()
    db.commit()

    return SimpleResponse(
        success=True,
        message="Scent Passport cleared. Ready for recalibration.",
    )
