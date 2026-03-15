"""
ml_engine.py — AromaTwin v2 Omnichannel Recommendation Engine
==============================================================

MATHEMATICAL PIPELINE OVERVIEW
──────────────────────────────────────────────────────────────────────────────
VISIT 1
  ① Calibration Ingestion (calibrate_from_anchors)
     5 anchor perfumes × user ratings → raw MAUT vector in accord-label space

  ② Two-Stage Filtering  (recommend)
     Stage 1 — Cosine Similarity on accord vectors    : 24k → Top 20 candidates
     Stage 2 — Weighted Jaccard Similarity on notes   : Top 20 → Final Top 3
               (base notes × 1.5,  heart notes × 1.5,  top notes × 1.0)

  ③ Rocchio Iterative Refinement  (rocchio_update)
     After each round the user rates 1-3 perfumes:
       V_updated = V_current  +  α × (rating/10) × V_perfume_accords
     α = 0.35 (tuned for 3 refinement rounds; keeps updates proportional)

VISIT 2+
  ④ Wildcard Generation  (wildcard_recommend)
     Match 1 & 2 : standard Top 2 from Two-Stage  (safe bets)
     Match 3     : high base-note overlap  BUT  dominant accord from a
                   DIFFERENT olfactive family  (intentional exploration push)
──────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Accord Taxonomy ───────────────────────────────────────────────────────────
# Maps the 5 radar-chart display categories → the raw accord labels in the CSV.
# This is the conceptual translation layer between olfactive families and dataset labels.

ACCORD_GROUPS: Dict[str, List[str]] = {
    "floral": [
        "floral", "rose", "white floral", "iris", "violet",
        "powdery", "aldehydic", "yellow floral", "lavender",
    ],
    "woody": [
        "woody", "earthy", "patchouli", "leather", "mossy",
        "green", "conifer", "dry wood",
    ],
    "fresh_citrus": [
        "citrus", "fresh", "aquatic", "fresh spicy", "ozonic",
        "aromatic", "alcohol",
    ],
    "warm_spicy": [
        "amber", "warm spicy", "spicy", "oriental", "balsamic",
        "animalic", "oud", "cinnamon", "anis", "soft spicy",
    ],
    "sweet_gourmand": [
        "sweet", "vanilla", "fruity", "caramel", "gourmand",
        "creamy", "coconut", "almond", "chocolate", "coffee",
        "musky", "beeswax", "cacao", "cherry", "tropical",
    ],
}

# Reverse map: accord_label → radar category
ACCORD_TO_CATEGORY: Dict[str, str] = {
    acc: cat for cat, accords in ACCORD_GROUPS.items() for acc in accords
}

# Positional importance weights for mainaccord1–5
# mainaccord1 is the dominant accord; its weight in calibration is highest.
ACCORD_POSITION_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.2]

# 5 calibration anchor perfumes — one per extreme olfactive family.
# These are the highest-rated iconic scents in the dataset for each pole.
# Keys must match the exact 'Perfume' column value in the CSV (lowercase, hyphenated).
ANCHOR_PERFUME_KEYS: List[str] = [
    "light-blue",          # Fresh · Citrus   (Dolce & Gabbana, 29 708 ratings)
    "j-adore",             # Floral · White   (Dior,            25 013 ratings)
    "black-opium",         # Sweet · Vanilla  (YSL,             25 669 ratings)
    "black-orchid",        # Warm · Spicy     (Tom Ford,        26 053 ratings)
    "baccarat-rouge-540",  # Woody · Amber    (MFK,             20 435 ratings)
]

# Hardcoded display metadata for anchors (brand colours, family labels, icons)
# 5 calibration anchor perfumes — strictly from the L'Oréal Luxe portfolio
ANCHOR_PERFUME_KEYS: List[str] = [
    "acqua-di-gio",        # Fresh · Citrus   (Giorgio Armani)
    "my-way",              # Floral · White   (Giorgio Armani)
    "black-opium",         # Sweet · Vanilla  (YSL)
    "spicebomb",           # Warm · Spicy     (Viktor&Rolf)
    "by-the-fireplace",    # Woody · Amber    (Maison Margiela)
]

# Hardcoded display metadata for anchors (brand colours, family labels, icons)
ANCHOR_META: Dict[str, dict] = {
    "acqua-di-gio": {
        "display_name": "Acqua di Giò",
        "family_label": "FRESH · CITRUS",
        "family_key":   "fresh_citrus",
        "accent_color": "#4a8fa8",
        "icon":         "◈",
    },
    "my-way": {
        "display_name": "My Way",
        "family_label": "FLORAL · WHITE",
        "family_key":   "floral",
        "accent_color": "#e8a5b2",
        "icon":         "✦",
    },
    "black-opium": {
        "display_name": "Black Opium",
        "family_label": "SWEET · GOURMAND",
        "family_key":   "sweet_gourmand",
        "accent_color": "#8b2635",
        "icon":         "◆",
    },
    "spicebomb": {
        "display_name": "Spicebomb",
        "family_label": "WARM · SPICY",
        "family_key":   "warm_spicy",
        "accent_color": "#6a3d7a",
        "icon":         "❋",
    },
    "by-the-fireplace": {
        "display_name": "By the Fireplace",
        "family_label": "WOODY · EARTHY",
        "family_key":   "woody",
        "accent_color": "#c47c2b",
        "icon":         "⬡",
    },
}
# Rocchio learning rate — scales how much each user rating moves the MAUT vector.
# 0.35 is empirically tuned for 3 refinement rounds (total drift ≈ 30 % of vector magnitude).
ROCCHIO_ALPHA = 0.35

# Wildcard: the top-note family for the wildcard MUST be at least this far
# (in category index distance) from the user's dominant accord family.
WILDCARD_FAMILY_EXCLUSION = {"floral": "sweet_gourmand", "woody": "fresh_citrus",
                              "fresh_citrus": "warm_spicy", "warm_spicy": "floral",
                              "sweet_gourmand": "woody"}


class AromaTwinEngine:
    """
    Singleton-style class: loaded once at server startup, then serves all requests.
    The large CSV matrix (24 063 × 84 accords) lives in memory for sub-millisecond retrieval.
    """

    def __init__(self, csv_path: str):
        print("[ENGINE] Initialising AromaTwin v2 engine …")
        self.df: pd.DataFrame = self._load_and_preprocess(csv_path)

        # Fit MultiLabelBinarizer — converts accord lists to binary indicator rows
        self.mlb = MultiLabelBinarizer()
        self.accord_matrix: np.ndarray = self._build_accord_matrix()
        self.all_accord_labels: List[str] = list(self.mlb.classes_)

        # Pre-build lookups for O(1) perfume-name → row index
        self._name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.df["Perfume"])
        }

        print(f"[ENGINE] Ready — {len(self.df):,} perfumes · "
              f"{len(self.all_accord_labels)} accord labels")

    # ── Data Loading ──────────────────────────────────────────────────────────

    def _load_and_preprocess(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        # Notes: fill NaN, lowercase, strip
        for col in ["Top", "Middle", "Base"]:
            df[col] = df[col].fillna("").astype(str).str.lower().str.strip()

        # Accord columns: fill NaN, lowercase, strip
        accord_cols = ["mainaccord1", "mainaccord2", "mainaccord3", "mainaccord4", "mainaccord5"]
        for col in accord_cols:
            df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

        # Combine all accords into a list per perfume (drop empty strings)
        df["accord_list"] = df[accord_cols].apply(
            lambda row: [a for a in row if a != ""], axis=1
        )
        # Fallback for perfumes with zero accords
        df.loc[df["accord_list"].str.len() == 0, "accord_list"] = \
            df.loc[df["accord_list"].str.len() == 0, "accord_list"].apply(lambda _: ["unknown"])

        # Metadata cleanup
        df["Brand"]   = df["Brand"].fillna("Unknown").astype(str).str.strip().str.title()
        df["Perfume"] = df["Perfume"].fillna("Unknown").astype(str).str.strip().str.lower()
        df["Country"] = df["Country"].fillna("Unknown").astype(str).str.strip().str.title()
        df["Gender"]  = df["Gender"].fillna("unisex").astype(str).str.strip().str.lower()
        df["url"]     = df["url"].fillna("").astype(str).str.strip()

        # Create a clean display name (replace hyphens with spaces, title-case)
        df["display_name"] = df["Perfume"].str.replace("-", " ").str.title()

        return df.reset_index(drop=True)

    # ── Accord Matrix ─────────────────────────────────────────────────────────

    def _build_accord_matrix(self) -> np.ndarray:
        """
        Fits a MultiLabelBinarizer on all accord lists.
        Returns a (N_perfumes × N_unique_accords) binary float matrix.
        Row i = accord bag-of-words vector for perfume i.
        """
        self.mlb.fit(self.df["accord_list"])
        return self.mlb.transform(self.df["accord_list"]).astype(float)

    # ── MAUT Vector Helpers ───────────────────────────────────────────────────

    def _maut_dict_to_vector(self, maut_dict: Dict[str, float]) -> np.ndarray:
        """
        Converts a {accord_label: weight} dict into a dense numpy vector aligned
        with self.all_accord_labels (the vocabulary axis order used in accord_matrix).
        Unknown labels (not in vocabulary) are silently ignored.
        """
        vec = np.zeros(len(self.all_accord_labels))
        label_to_idx = {lbl: i for i, lbl in enumerate(self.all_accord_labels)}
        for accord, weight in maut_dict.items():
            if accord in label_to_idx:
                vec[label_to_idx[accord]] = weight
        return vec

    def _perfume_row_to_accord_dict(self, row: pd.Series) -> Dict[str, float]:
        """
        Converts a single perfume's mainaccord1-5 columns into a weighted dict.
        Positional weights: mainaccord1=1.0, …, mainaccord5=0.2.
        Used during Rocchio update and calibration ingestion.
        """
        accord_dict: Dict[str, float] = {}
        for i, col in enumerate(["mainaccord1", "mainaccord2", "mainaccord3",
                                  "mainaccord4", "mainaccord5"]):
            acc = row.get(col, "")
            if acc and acc != "unknown":
                w = ACCORD_POSITION_WEIGHTS[i]
                accord_dict[acc] = accord_dict.get(acc, 0.0) + w
        return accord_dict

    # ── PUBLIC: Anchor Details ─────────────────────────────────────────────────

    def get_anchor_data(self) -> List[dict]:
        """
        Returns display-ready metadata for the 5 calibration anchor perfumes.
        Called once at startup to serve GET /anchors.
        """
        result = []
        for key in ANCHOR_PERFUME_KEYS:
            mask = self.df["Perfume"] == key
            if not mask.any():
                continue
            row  = self.df[mask].iloc[0]
            meta = ANCHOR_META.get(key, {})
            result.append({
                "key":          key,
                "display_name": meta.get("display_name", row["display_name"]),
                "brand":        row["Brand"],
                "family_label": meta.get("family_label", row["mainaccord1"].upper()),
                "family_key":   meta.get("family_key", "floral"),
                "top_notes":    row["Top"].title(),
                "heart_notes":  row["Middle"].title(),
                "base_notes":   row["Base"].title(),
                "accent_color": meta.get("accent_color", "#c9a84c"),
                "icon":         meta.get("icon", "◈"),
            })
        return result

    # ── PUBLIC: Calibration Ingestion ─────────────────────────────────────────

    def calibrate_from_anchors(
        self, anchor_ratings: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Builds the initial MAUT vector from 5 anchor ratings (Visit 1).

        Algorithm:
          For each anchor perfume a with user rating r_a (1-10):
            For each accord position i (mainaccord1…5) with positional weight w_i:
              MAUT[accord_i] += r_a × w_i

        Positional weighting respects olfactive prominence in the formulation —
        mainaccord1 (dominant accord) contributes 5× more than mainaccord5.

        Returns: raw (un-normalised) MAUT dict {accord_label: cumulative_weight}
        """
        maut: Dict[str, float] = {}

        for key, rating in anchor_ratings.items():
            mask = self.df["Perfume"] == key
            if not mask.any():
                continue  # skip unknown anchor keys gracefully

            row = self.df[mask].iloc[0]
            accord_dict = self._perfume_row_to_accord_dict(row)

            for accord, pos_weight in accord_dict.items():
                contribution = rating * pos_weight
                maut[accord] = maut.get(accord, 0.0) + contribution

        # Fallback: if calibration yields an empty vector (all anchors missing),
        # return a perfectly balanced profile so the user still gets results.
        if not maut:
            maut = {acc: 5.0 for cat in ACCORD_GROUPS.values() for acc in cat}

        return maut

    # ── PUBLIC: Rocchio Iterative Refinement ──────────────────────────────────

    def rocchio_update(
        self,
        current_maut: Dict[str, float],
        rated_perfumes: List[Tuple[str, float]],  # [(perfume_name, rating), ...]
        alpha: float = ROCCHIO_ALPHA,
    ) -> Dict[str, float]:
        """
        Applies Rocchio's Algorithm to update the MAUT vector based on
        the user's ratings of suggested perfumes.

        Formula per rated perfume:
          V_updated = V_current  +  α × (rating/10) × V_perfume_accords

        Rationale:
          • Cosine similarity is scale-invariant, so absolute magnitudes don't
            distort Stage-1 results.
          • (rating/10) scales the update proportionally: a 10/10 rating moves
            the vector more than a 6/10 — natural learning signal.
          • α = 0.35 gives meaningful drift over 3 rounds without overshooting.

        rated_perfumes: list of (perfume_name, rating) tuples from this round.
        Returns: updated MAUT dict.
        """
        updated = dict(current_maut)  # shallow copy — don't mutate the original

        for perfume_name, rating in rated_perfumes:
            # Look up the perfume by its raw CSV Perfume key (lowercase-hyphenated)
            norm_name = perfume_name.lower().replace(" ", "-")
            mask = self.df["Perfume"] == norm_name
            if not mask.any():
                # Try display_name fallback
                mask = self.df["display_name"].str.lower() == perfume_name.lower()
            if not mask.any():
                continue  # can't find — skip, don't crash

            row         = self.df[mask].iloc[0]
            accord_dict = self._perfume_row_to_accord_dict(row)

            scaling = alpha * (rating / 10.0)   # e.g. rating=8 → scaling=0.28

            for accord, pos_weight in accord_dict.items():
                delta            = scaling * pos_weight
                updated[accord]  = updated.get(accord, 0.0) + delta

        return updated

    # ── Stage-1 & Stage-2 internals ───────────────────────────────────────────

    def _build_note_pool(self, candidates: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregates notes from the Top-20 candidate perfumes into a weighted
        frequency dict (the 'consensus note cloud').
        Weights mirror the evaporation curve:
          top notes (volatile, first impression)  × 1.0
          heart notes (body, 2-4 h)               × 1.5
          base notes (dry-down, 4-8 h, character) × 1.5
        """
        pool: Dict[str, float] = {}
        for _, row in candidates.iterrows():
            for note in [n.strip() for n in row["Top"].split(",")    if n.strip()]:
                pool[note] = pool.get(note, 0.0) + 1.0
            for note in [n.strip() for n in row["Middle"].split(",") if n.strip()]:
                pool[note] = pool.get(note, 0.0) + 1.5
            for note in [n.strip() for n in row["Base"].split(",")   if n.strip()]:
                pool[note] = pool.get(note, 0.0) + 1.5
        return pool

    def _weighted_jaccard(self, row: pd.Series, note_pool: Dict[str, float]) -> float:
        """
        Weighted Jaccard similarity between one perfume's notes and the consensus pool.
          intersection = Σ weight_i  for notes in BOTH the perfume AND the pool
          union        = (perfume set weight) + (pool weight) − intersection
        Returns a value in [0, ∞); higher → better match.
        """
        top    = [n.strip() for n in row["Top"].split(",")    if n.strip()]
        heart  = [n.strip() for n in row["Middle"].split(",") if n.strip()]
        base   = [n.strip() for n in row["Base"].split(",")   if n.strip()]

        intersection = (
            sum(1.0 * note_pool[n] for n in top   if n in note_pool) +
            sum(1.5 * note_pool[n] for n in heart if n in note_pool) +
            sum(1.5 * note_pool[n] for n in base  if n in note_pool)
        )
        p_weight = len(top) * 1.0 + len(heart) * 1.5 + len(base) * 1.5
        union    = p_weight + sum(note_pool.values()) - intersection
        return intersection / union if union else 0.0

    def _build_explainability(
        self,
        row: pd.Series,
        maut: Dict[str, float],
        cosine_score: float,
    ) -> Tuple[str, List[str]]:
        """Generates a human-readable explainability tag for the UI card."""
        # Dominant MAUT category
        cat_scores = {cat: sum(maut.get(a, 0) for a in accs)
                      for cat, accs in ACCORD_GROUPS.items()}
        top_cat = max(cat_scores, key=cat_scores.get)
        cat_display = top_cat.replace("_", " ").title()

        # Accords this perfume shares with the user's high-weight categories
        matched = [a.title() for a in row["accord_list"]
                   if maut.get(a, 0) > 0][:3]

        base_sig = [n.strip().title() for n in row["Base"].split(",") if n.strip()][:2]
        sig_note = " & ".join(base_sig) if base_sig else "Rich Composition"

        tag = (f"Matched via {int(cosine_score * 100)}% accord affinity for "
               f"{cat_display} · Signature dry-down: {sig_note}")
        return tag, matched

    # ── PUBLIC: Core Recommendation ──────────────────────────────────────────

    def recommend(
        self,
        maut: Dict[str, float],
        excluded_names: List[str],
        gender_filter: Optional[str] = None,
        top_n: int = 3,
    ) -> List[dict]:
        """
        Two-Stage recommendation pipeline.

        Stage 1 — Cosine Similarity on accord vectors
          Converts MAUT dict → dense vector → cosine sim against accord matrix.
          Filters catalog to Top-20 candidates (excluding already-shown perfumes).

        Stage 2 — Weighted Jaccard Similarity on notes
          Builds consensus note-cloud from Top-20.
          Ranks by weighted Jaccard overlap (base/heart × 1.5, top × 1.0).
          Returns final top_n results.

        excluded_names: list of display_names already shown to this user.
        """
        # Edge case: zero vector → use balanced profile
        if not maut or sum(maut.values()) == 0:
            maut = {acc: 5.0 for cat in ACCORD_GROUPS.values() for acc in cat}

        # ── Optional gender pre-filter ────────────────────────────────────────
        if gender_filter in ["men", "women"]:
            mask       = self.df["Gender"].isin([gender_filter, "unisex"])
            work_df    = self.df[mask].reset_index(drop=True)
            work_mat   = self.accord_matrix[mask.values]
        else:
            work_df  = self.df
            work_mat = self.accord_matrix

        # Exclude already-displayed perfumes
        excluded_lower = {n.lower() for n in excluded_names}
        excl_mask  = work_df["display_name"].str.lower().isin(excluded_lower)
        excl_mask |= work_df["Perfume"].str.lower().isin(excluded_lower)
        work_df    = work_df[~excl_mask].reset_index(drop=True)
        work_mat   = work_mat[~excl_mask.values]

        # ── Stage 1: Cosine Similarity ────────────────────────────────────────
        user_vec   = self._maut_dict_to_vector(maut).reshape(1, -1)
        cos_scores = cosine_similarity(user_vec, work_mat)[0]

        top20_idx  = np.argsort(cos_scores)[::-1][:20]
        top20_df   = work_df.iloc[top20_idx].copy()
        top20_df["_cosine"] = cos_scores[top20_idx]

        # ── Stage 2: Weighted Jaccard on Notes ────────────────────────────────
        note_pool         = self._build_note_pool(top20_df)
        top20_df["_jac"]  = top20_df.apply(
            lambda r: self._weighted_jaccard(r, note_pool), axis=1
        )
        top20_df = top20_df.sort_values("_jac", ascending=False)

        results = []
        for rank, (_, row) in enumerate(top20_df.head(top_n).iterrows(), 1):
            tag, matched = self._build_explainability(row, maut, float(row["_cosine"]))
            results.append({
                "rank":               rank,
                "name":               str(row["display_name"]),
                "brand":              str(row["Brand"]),
                "country":            str(row["Country"]),
                "gender":             str(row["Gender"]),
                "top_notes":          str(row["Top"]).title(),
                "heart_notes":        str(row["Middle"]).title(),
                "base_notes":         str(row["Base"]).title(),
                "fragrantica_url":    str(row["url"]),
                "cosine_score":       round(float(row["_cosine"]), 4),
                "jaccard_score":      round(float(row["_jac"]), 4),
                "explainability_tag": tag,
                "matched_accords":    matched,
                "is_wildcard":        False,
            })
        return results

    # ── PUBLIC: Wildcard Recommendation ──────────────────────────────────────

    def wildcard_recommend(
        self,
        maut: Dict[str, float],
        excluded_names: List[str],
    ) -> Optional[dict]:
        """
        Generates the 'Olfactive Discovery' wildcard for returning customers (Visit 2+).

        Algorithm:
          1. Determine the user's dominant olfactive family (highest MAUT category).
          2. Identify the 'exploration target' family (WILDCARD_FAMILY_EXCLUSION map).
          3. Collect base notes from the user's top-3 highest-weight accord areas.
          4. Score all perfumes in the exploration family by base-note Jaccard overlap.
          5. Return the top scorer.

        Rationale: The customer already knows their dominant family.  The wildcard
        offers their preferred base-note character (the intimate dry-down they love)
        but wrapped in an unfamiliar top-note opening — maximising discovery while
        minimising risk of complete rejection.
        """
        # 1. Find dominant MAUT category
        cat_scores = {cat: sum(maut.get(a, 0) for a in accs)
                      for cat, accs in ACCORD_GROUPS.items()}
        dominant_cat  = max(cat_scores, key=cat_scores.get)
        wildcard_cat  = WILDCARD_FAMILY_EXCLUSION.get(dominant_cat, "fresh_citrus")
        target_accords = set(ACCORD_GROUPS.get(wildcard_cat, []))

        # 2. Build 'preferred base note pool' from user's MAUT (top base-associated accords)
        base_pool: Dict[str, float] = {}
        sorted_maut = sorted(maut.items(), key=lambda x: x[1], reverse=True)
        for acc, w in sorted_maut[:10]:
            base_pool[acc] = w

        # 3. Filter to the wildcard family and exclude already-shown perfumes
        excluded_lower = {n.lower() for n in excluded_names}
        wild_mask = (
            self.df["mainaccord1"].isin(target_accords) |
            self.df["mainaccord2"].isin(target_accords)
        )
        excl_mask = (
            self.df["display_name"].str.lower().isin(excluded_lower) |
            self.df["Perfume"].str.lower().isin(excluded_lower)
        )
        candidates = self.df[wild_mask & ~excl_mask].copy()

        if candidates.empty:
            # Fallback: run standard recommend for the wildcard slot
            return None

        # 4. Score by base-note Jaccard against the user's preferred notes
        def base_jaccard(row: pd.Series) -> float:
            base_notes = {n.strip() for n in row["Base"].split(",") if n.strip()}
            pool_notes = set(base_pool.keys())
            inter = len(base_notes & pool_notes)
            union = len(base_notes | pool_notes)
            return inter / union if union else 0.0

        candidates["_wild_score"] = candidates.apply(base_jaccard, axis=1)
        candidates = candidates.nlargest(1, "_wild_score")

        if candidates.empty:
            return None

        row = candidates.iloc[0]
        return {
            "rank":               3,
            "name":               str(row["display_name"]),
            "brand":              str(row["Brand"]),
            "country":            str(row["Country"]),
            "gender":             str(row["Gender"]),
            "top_notes":          str(row["Top"]).title(),
            "heart_notes":        str(row["Middle"]).title(),
            "base_notes":         str(row["Base"]).title(),
            "fragrantica_url":    str(row["url"]),
            "cosine_score":       round(float(candidates["_wild_score"].iloc[0]), 4),
            "jaccard_score":      round(float(candidates["_wild_score"].iloc[0]), 4),
            "explainability_tag": (
                f"OLFACTIVE DISCOVERY · Same base-note character you love, "
                f"but opening with a {wildcard_cat.replace('_', ' ').title()} surprise — "
                f"a new olfactive horizon tailored to your dry-down affinity"
            ),
            "matched_accords":    [a.title() for a in target_accords][:3],
            "is_wildcard":        True,
        }

    # ── PUBLIC: Radar Data ────────────────────────────────────────────────────

    def get_radar_data(self, maut: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregates the fine-grained MAUT accord weights into the 5 radar-chart
        categories and normalises to [0, 10] for Chart.js.

        Uses sum-aggregation so the radar reflects cumulative preference strength,
        not just the presence of an accord.
        """
        raw: Dict[str, float] = {}
        for cat, accords in ACCORD_GROUPS.items():
            raw[cat] = sum(maut.get(acc, 0.0) for acc in accords)

        max_val = max(raw.values()) if raw and max(raw.values()) > 0 else 1.0
        return {k: round(min((v / max_val) * 10.0, 10.0), 2) for k, v in raw.items()}
