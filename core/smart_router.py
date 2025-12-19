# core/smart_router.py

import re
from .intent_classifier import classify_query_intent
from .lifestyle_rag_v2 import try_build_lifestyle_card
from .lifestyle_rag_v2 import _looks_like_lifestyle_query

# Clustering (safe to import at top)
from .cluster_router import (
    cluster_all,
    cluster_by_state,
    cluster_single_city,
    cluster_similar_to,
)

# ✅ ADD THIS — ML Ranking imports
from .ml_router import (
    run_family_ranking,
    run_young_ranking,
    run_retirement_ranking,
    run_single_city_prediction,
)

# -------------------------------------------------------------------
# Helper: extract city name from natural language
# -------------------------------------------------------------------
def extract_city_name(q: str):
    """
    Extract a clean city name from natural language questions.

    Examples:
        "Which cluster is Miami in"   -> "Miami"
        "cluster of Austin"           -> "Austin"
        "cities like Chicago"         -> "Chicago"
        "similar cities to Dallas"    -> "Dallas"
    """
    q = q.lower().strip()
    q = q.replace("?", "").replace(".", "")

    # Remove common routing terms
    filter_words = {
        "cluster", "clusters", "cities", "city",
        "in", "of", "for", "is", "to", "similar",
        "like", "best", "which", "what"
    }

    tokens = [t for t in q.split() if t not in filter_words]
    if not tokens:
        return None

    city = tokens[-1].strip().title()
    return city if len(city) > 1 else None


# -------------------------------------------------------------------
# Helper: decide if user wants a SINGLE best city or a LIST
# -------------------------------------------------------------------
def _is_single_city_question(q: str) -> bool:
    q = q.lower()
    single_phrases = [
        "which city",
        "what city",
        "best city",
        "single best city",
        "no of city 1",
    ]
    return any(p in q for p in single_phrases)


def _is_family_intent(q: str) -> bool:
    q = q.lower()
    return ("family" in q) or ("families" in q)


def _is_young_intent(q: str) -> bool:
    q = q.lower()
    return ("young professional" in q) or ("young professionals" in q) or ("young people" in q)


def _is_retirement_intent(q: str) -> bool:
    q = q.lower()
    return ("retire" in q) or ("retirement" in q) or ("senior" in q) or ("seniors" in q)


# -------------------------------------------------------------------
# MASTER ROUTER
# -------------------------------------------------------------------
def smart_route(query: str):
    """
    Master router:
      - ML ranking (families, young, retirement)
      - single-city ML scoring
      - clustering (all, state, single, similar)
      - fallback for SQL/semantic
        
    Returns:
        (mode, payload)
    """

    q = query.lower().strip()
    single_best = _is_single_city_question(q)

    # -------------------------------------------------------
    # 🔥 LIFESTYLE RAG — Detect lifestyle queries first
    # -------------------------------------------------------
    if _looks_like_lifestyle_query(q):
        lifestyle_card = try_build_lifestyle_card(query)
        if lifestyle_card:
            return "lifestyle_card", lifestyle_card


    # -------------------------------------------------------
    # 🔥 ML RANKING: Families
    # -------------------------------------------------------
    if _is_family_intent(q) and "best" in q:
        mode_intent, state = classify_query_intent(query)
        df_rank = run_family_ranking(state)

        if single_best and not df_rank.empty:
            return "ml_family_single", df_rank.head(1)

        return "ml_family_list", df_rank

    # -------------------------------------------------------
    # 🔥 ML RANKING: Young Professionals
    # -------------------------------------------------------
    if _is_young_intent(q) and "best" in q:
        mode_intent, state = classify_query_intent(query)
        df_rank = run_young_ranking(state)

        if single_best and not df_rank.empty:
            return "ml_young_single", df_rank.head(1)

        return "ml_young_list", df_rank

    # -------------------------------------------------------
    # 🔥 ML RANKING: Retirement
    # -------------------------------------------------------
    if _is_retirement_intent(q) and "best" in q:
        mode_intent, state = classify_query_intent(query)
        df_rank = run_retirement_ranking(state)

        if single_best and not df_rank.empty:
            return "ml_retirement_single", df_rank.head(1)

        return "ml_retirement_list", df_rank

    # -------------------------------------------------------
    # 🔥 ML — Single-city prediction
    # -------------------------------------------------------
    if "score for" in q or "predict" in q:
        city = extract_city_name(q)
        if city:
            info = run_single_city_prediction(city)
            if info:
                return "ml_single_city", info

    # -------------------------------------------------------
    # 🔥 CLUSTERING — All cities
    # -------------------------------------------------------
    if (
        "cluster all" in q
        or "all clusters" in q
        or "cluster every city" in q
        or q.strip() == "cluster all cities"
    ):
        return "cluster_all", cluster_all()

    # -------------------------------------------------------
    # 🔥 CLUSTERING — Cities in a state
    # -------------------------------------------------------
    if ("cluster" in q or "group" in q) and "in" in q:
        parts = q.split("in")
        state = parts[-1].strip().title()
        if state:
            return "cluster_state", cluster_by_state(state)

    # -------------------------------------------------------
    # 🔥 CLUSTERING — Which cluster is <CITY> in?
    # -------------------------------------------------------
    if (
        "which cluster" in q
        or ("cluster" in q and ("in" in q or "of" in q or "for" in q))
    ):
        city = extract_city_name(q)
        if city:
            return "cluster_single", cluster_single_city(city)

    # -------------------------------------------------------
    # 🔥 CLUSTERING — Similar cities
    # -------------------------------------------------------
    if "similar" in q or "like" in q:
        city = extract_city_name(q)
        if city:
            return "cluster_similar", cluster_similar_to(city)

    # -------------------------------------------------------
    # ❗ FALLBACK → SQL or semantic search
    # -------------------------------------------------------
    return None, None
