"""
CitySearch AI - Smart Classifier with Verification Pipeline
=============================================================
Industry-standard hybrid classification with LLM verification.

Pipeline:
1. Initial Classification (LLM or Rule-Based)
2. Verification Layer (LLM confirms/corrects)
3. Confidence Check (reinterpret if needed)
4. Final Classification Output
"""

import streamlit as st
import json
from typing import Optional, Dict, Any, Tuple


 
# CONFIGURATION
 
CLASSIFIER_MODEL = "gpt-4o-mini"
VERIFIER_MODEL = "gpt-4o-mini"
MAX_REINTERPRET_ATTEMPTS = 2
CONFIDENCE_THRESHOLD = 0.7


 
# MAIN ENTRY POINT
 
def smart_classify(query: str) -> Dict[str, Any]:
    """
    Smart classification with verification pipeline.
    
    Returns a verified, high-confidence classification.
    """
    
    # Stage 1: Initial Classification
    initial = _stage1_initial_classification(query)
    
    # Stage 2: Verification
    verified = _stage2_verify_classification(query, initial)
    
    # Stage 3: Confidence Check & Reinterpret if needed
    final = _stage3_confidence_check(query, verified)

    # NEW: normalize state & city filters for router
    entities = final.get("entities", {})
    states = entities.get("states", [])
    cities = entities.get("cities", [])
    
    final["state_filter"] = states[0] if states else None
    final["city_filter"] = cities[0] if cities else None

    # Add metadata
    final["pipeline_complete"] = True
    final["source"] = final.get("source", "smart_classifier")
    
    return final


 
# STAGE 1: INITIAL CLASSIFICATION
 
def _stage1_initial_classification(query: str) -> Dict[str, Any]:
    """
    Stage 1: Get initial classification from LLM.
    Falls back to rule-based if LLM fails.
    """
    
    # Try LLM first
    llm_result = _llm_initial_classify(query)
    
    if llm_result and llm_result.get("success"):
        llm_result["classification_source"] = "llm"
        return llm_result
    
    # Fallback to rule-based
    rule_result = _rule_based_classify(query)
    rule_result["classification_source"] = "rule_based"
    return rule_result


def _llm_initial_classify(query: str) -> Optional[Dict[str, Any]]:
    """LLM initial classification."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{
                "role": "system",
                "content": """You are a query classifier for a US cities database.

DATABASE COLUMNS: city, state, population, median_age, avg_household_size, state_code

CLASSIFICATION RULES:
- "population of [city]" → single_city with metric
- "population of [state]" → single_state with metric  
- "top N [metric] cities" → superlative with limit
- "which city has highest/lowest X" → superlative
- "best cities for families/professionals/retirement" → ranking with intent
- "compare X and Y" → comparison
- "how many cities in [state]" → aggregate
- "life in [city]" → lifestyle
- "cities with population > X" → filtered_list
- General city lists → city_list

Return JSON:
{
    "query_type": "single_city" | "single_state" | "superlative" | "comparison" | "ranking" | "aggregate" | "lifestyle" | "filtered_list" | "city_list",
    "entities": {
        "cities": ["list of city names"],
        "states": ["list of state names"]
    },
    "metric": "population" | "median_age" | "avg_household_size" | null,
    "direction": "highest" | "lowest" | null,
    "limit": number | null,
    "filter": {
        "column": "column_name" | null,
        "operator": ">" | "<" | "=" | ">=" | "<=" | null,
        "value": number | null
    },
    "intent": "families" | "young_professionals" | "retirement" | "general",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation of classification"
}"""
            }, {
                "role": "user",
                "content": query
            }],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0
        )
        
        result = json.loads(response.choices[0].message.content)
        result["success"] = True
        
        # NEW: normalize sort_direction for router
        if "direction" in result:
            result["sort_direction"] = result["direction"]
        
        return result

        
    except Exception as e:
        print(f"LLM classification error: {e}")
        return None


def _rule_based_classify(query: str) -> Dict[str, Any]:
    """Rule-based fallback classification."""
    import re
    
    q = query.lower().strip()
    result = {
        "success": True,
        "query_type": "city_list",
        "entities": {"cities": [], "states": []},
        "metric": None,
        "direction": None,
        "limit": 10,
        "filter": {"column": None, "operator": None, "value": None},
        "intent": "general",
        "confidence": 0.5,
        "reasoning": "Rule-based fallback classification"
    }
    
    # Detect states
    us_states = [
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york",
        "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
        "pennsylvania", "rhode island", "south carolina", "south dakota",
        "tennessee", "texas", "utah", "vermont", "virginia", "washington",
        "west virginia", "wisconsin", "wyoming"
    ]
    
    for state in us_states:
        if state in q:
            result["entities"]["states"].append(state.title())
    
    # Detect metrics
    if "population" in q:
        result["metric"] = "population"
    elif "median age" in q or "age" in q:
        result["metric"] = "median_age"
    elif "household" in q:
        result["metric"] = "avg_household_size"
    
    # Detect direction
    if any(w in q for w in ["highest", "largest", "most", "biggest", "top"]):
        result["direction"] = "highest"
    elif any(w in q for w in ["lowest", "smallest", "least", "bottom"]):
        result["direction"] = "lowest"
    
    # Detect limit
    limit_match = re.search(r"top\s+(\d+)", q)
    if limit_match:
        result["limit"] = int(limit_match.group(1))
    
    # Detect query type
    if "compare" in q or " vs " in q or " versus " in q:
        result["query_type"] = "comparison"
        result["confidence"] = 0.7
    elif any(w in q for w in ["highest", "largest", "lowest", "smallest"]) and result["metric"]:
        result["query_type"] = "superlative"
        result["confidence"] = 0.7
    elif "how many" in q or "count" in q:
        result["query_type"] = "aggregate"
        result["confidence"] = 0.7
    elif "best" in q:
        result["query_type"] = "ranking"
        if any(w in q for w in ["family", "families", "kids", "children"]):
            result["intent"] = "families"
        elif any(w in q for w in ["young", "professional", "adult"]):
            result["intent"] = "young_professionals"
        elif any(w in q for w in ["retire", "retirement", "senior"]):
            result["intent"] = "retirement"
        result["confidence"] = 0.6
    elif "life in" in q or "living in" in q:
        result["query_type"] = "lifestyle"
        result["confidence"] = 0.8
    elif result["entities"]["states"] and result["metric"]:
        result["query_type"] = "single_state"
        result["confidence"] = 0.6

    # NEW: normalize sort_direction for router
    result["sort_direction"] = result.get("direction")
    return result


 
# STAGE 2: VERIFICATION LAYER
 
def _stage2_verify_classification(query: str, initial: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2: LLM verifies the initial classification.
    Can confirm, correct, or flag for reinterpretation.
    """
    
    # Skip verification if initial confidence is very high
    if initial.get("confidence", 0) >= 0.95:
        initial["verified"] = True
        initial["verification_action"] = "skipped_high_confidence"
        return initial
    
    verification = _llm_verify(query, initial)
    
    if verification is None:
        # Verification failed, use initial
        initial["verified"] = False
        initial["verification_action"] = "verification_failed"
        return initial
    
    if verification.get("is_correct"):
        initial["verified"] = True
        initial["verification_action"] = "confirmed"
        # Boost confidence since LLM confirmed
        initial["confidence"] = min(1.0, initial.get("confidence", 0.5) + 0.2)
        return initial
    else:
        # Use corrected classification
        corrected = verification.get("corrected_classification", initial)
        corrected["verified"] = True
        corrected["verification_action"] = "corrected"
        corrected["original_classification"] = initial
        corrected["correction_reason"] = verification.get("correction_reason", "")
        return corrected


def _llm_verify(query: str, classification: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """LLM verification of classification."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=VERIFIER_MODEL,
            messages=[{
                "role": "system",
                "content": """You are a verification system for query classification.

Your job is to check if the classification is correct for the user's query.

VERIFY these aspects:
1. Is the query_type correct?
2. Are the entities (cities/states) correctly identified?
3. Is the metric correct?
4. Is the direction (highest/lowest) correct?
5. Is the limit correct?
6. Is the intent correct?

Return JSON:
{
    "is_correct": true | false,
    "confidence": 0.0 to 1.0,
    "issues": ["list of issues if any"],
    "correction_reason": "why correction is needed" | null,
    "corrected_classification": { full corrected classification if is_correct=false } | null
}"""
            }, {
                "role": "user",
                "content": f"""QUERY: "{query}"

CLASSIFICATION:
{json.dumps(classification, indent=2)}

Is this classification correct?"""
            }],
            response_format={"type": "json_object"},
            max_tokens=400,
            temperature=0
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Verification error: {e}")
        return None


 
# STAGE 3: CONFIDENCE CHECK & REINTERPRET
 
def _stage3_confidence_check(query: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 3: Check confidence and reinterpret if needed.
    """
    
    confidence = classification.get("confidence", 0.5)
    
    if confidence >= CONFIDENCE_THRESHOLD:
        classification["reinterpreted"] = False
        return classification
    
    # Low confidence - try to reinterpret
    reinterpreted = _llm_reinterpret(query, classification)
    
    if reinterpreted and reinterpreted.get("confidence", 0) > confidence:
        reinterpreted["reinterpreted"] = True
        reinterpreted["original_low_confidence"] = classification
        return reinterpreted
    
    # Reinterpretation didn't help, use original
    classification["reinterpreted"] = False
    classification["reinterpret_attempted"] = True
    return classification


def _llm_reinterpret(query: str, low_confidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """LLM reinterpretation for low-confidence classifications."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[{
                "role": "system",
                "content": """The previous classification had low confidence. 
                
Carefully analyze the query and provide a better classification.

Focus on:
1. What is the user REALLY asking for?
2. What type of response would best answer their question?
3. Are there any ambiguities that need resolution?

Return the same JSON format as before with higher confidence if possible."""
            }, {
                "role": "user",
                "content": f"""QUERY: "{query}"

PREVIOUS LOW-CONFIDENCE CLASSIFICATION:
{json.dumps(low_confidence, indent=2)}

Please provide a better interpretation."""
            }],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        result["success"] = True
        return result
        
    except Exception as e:
        print(f"Reinterpretation error: {e}")
        return None


 
# UTILITY FUNCTIONS
 
def is_city_related(query: str) -> bool:
    """Quick check if query is about US cities."""
    city_keywords = [
        "city", "cities", "town", "population", "state", "states",
        "live", "living", "move", "moving", "best", "top", "largest",
        "smallest", "compare", "vs", "versus", "family", "families",
        "retire", "retirement", "young", "professional", "median age",
        "household"
    ]
    
    us_states = [
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "texas", "florida", "new york", "illinois", "pennsylvania", "ohio"
        # ... abbreviated for space
    ]
    
    q_lower = query.lower()
    
    if any(kw in q_lower for kw in city_keywords):
        return True
    if any(state in q_lower for state in us_states):
        return True
    
    return False


def get_classification_summary(classification: Dict[str, Any]) -> str:
    """Get human-readable summary of classification."""
    query_type = classification.get("query_type", "unknown")
    confidence = classification.get("confidence", 0)
    source = classification.get("classification_source", "unknown")
    verified = classification.get("verified", False)
    
    status = "✅ Verified" if verified else "⚠️ Unverified"
    
    return f"{query_type} | Confidence: {confidence:.0%} | Source: {source} | {status}"


 
# DEBUG / LOGGING
 
def debug_classification(query: str, show_in_ui: bool = False):
    """Run classification with full debug output."""
    
    result = smart_classify(query)
    
    debug_info = {
        "query": query,
        "final_classification": result.get("query_type"),
        "confidence": result.get("confidence"),
        "source": result.get("classification_source"),
        "verified": result.get("verified"),
        "verification_action": result.get("verification_action"),
        "reinterpreted": result.get("reinterpreted"),
        "entities": result.get("entities"),
        "metric": result.get("metric"),
        "direction": result.get("direction"),
        "limit": result.get("limit"),
        "intent": result.get("intent")
    }
    
    if show_in_ui:
        st.json(debug_info)
    
    return result, debug_info
