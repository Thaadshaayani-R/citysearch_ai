#query_router.py

from core.nlp_to_sql import build_sql as build_rule_based_sql
from core.gpt_fallback import generate_sql_with_gpt


def build_sql_with_fallback(user_query: str, use_gpt: bool = True) -> str:
    """
    Hybrid Text-to-SQL:
    1) Try rule-based engine first.
    2) If we think the query is too complex or user wants 'ai mode', use GPT fallback.
    """

    query_lower = user_query.lower()

    # If user explicitly asks to "use AI" or "use GPT", go straight to GPT
    if use_gpt and ("use ai" in query_lower or "use gpt" in query_lower):
        return generate_sql_with_gpt(user_query)

    # Simple heuristic: if query contains words like 'compare', 'explain', 'vs',
    # rule-based may not be enough → prefer GPT.
    complex_keywords = ["compare", "versus", "vs", "correlation", "relationship"]
    if use_gpt and any(k in query_lower for k in complex_keywords):
        return generate_sql_with_gpt(user_query)

    # Default: try rule-based first
    try:
        sql = build_rule_based_sql(user_query)
        return sql
    except Exception:
        # If rule-based fails for any reason → GPT fallback
        if use_gpt:
            return generate_sql_with_gpt(user_query)
        raise
