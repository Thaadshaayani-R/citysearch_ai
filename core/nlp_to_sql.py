#nlp_to_sql.py

import re

# These are your real table columns
SELECT_COLUMNS = [
    "city",
    "state",
    "population",
    "median_age",
    "avg_household_size",
    "state_code",
]

# Used for safety
VALID_COLUMNS = set(SELECT_COLUMNS)

MAX_LIMIT = 50  # limit rows returned

SORTABLE_COLUMNS = {
    "population": "population",
    "median age": "median_age",
    "median_age": "median_age",
    "age": "median_age",
    "household size": "avg_household_size",
    "avg household size": "avg_household_size",
    "average household size": "avg_household_size",
    "avg_household_size": "avg_household_size",
}

US_STATES = [
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware",
    "florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas","kentucky",
    "louisiana","maine","maryland","massachusetts","michigan","minnesota","mississippi",
    "missouri","montana","nebraska","nevada","new hampshire","new jersey","new mexico",
    "new york","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania",
    "rhode island","south carolina","south dakota","tennessee","texas","utah","vermont",
    "virginia","washington","west virginia","wisconsin","wyoming"
]


 
# BASIC HELPERS
 

def extract_limit(text: str):
    match = re.search(r"(top|first)\s+(\d+)", text, re.I)
    if match:
        return min(int(match.group(2)), MAX_LIMIT)
    return 10


def extract_numeric_filter(text: str):
    text = text.lower()

    patterns = [
        r"(population)\s*(>=|<=|>|<|=)\s*([0-9]+)",
        r"(population)\s*(over|above|greater than)\s*([0-9]+)",
        r"(median age|median_age)\s*(>=|<=|>|<|=)\s*([0-9]+)",
        r"(median age|median_age)\s*(over|above|greater than)\s*([0-9]+)",
        r"(avg household size|average household size|avg_household_size)\s*(>=|<=|>|<|=)\s*([0-9]+)",
        r"(avg household size|average household size|avg_household_size)\s*(over|above|greater than)\s*([0-9]+)"
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            col = m.group(1).replace(" ", "_")
            op = m.group(2)
            if op in ["over", "above", "greater than"]:
                op = ">"
            val = m.group(3)
            return f"{col} {op} {val}"

    return None


def extract_state_filter(text: str):
    text_lower = text.lower()
    for state in US_STATES:
        if state in text_lower:
            formatted = state.title()
            return f"state = '{formatted}'"
    return None


 
# COUNT QUERIES
 

def is_count_query(text: str):
    count_patterns = [
        r"how many",
        r"number of",
        r"count of",
        r"total cities",
        r"total number of",
    ]
    return any(re.search(x, text, re.I) for x in count_patterns)


def build_count_sql(text: str):
    state = extract_state_filter(text)
    numeric = extract_numeric_filter(text)

    where = []
    if state: where.append(state)
    if numeric: where.append(numeric)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    return f"""
    SELECT COUNT(*) AS total_cities
    FROM dbo.cities
    {where_sql};
    """.strip()


 
# SORTING
 

def detect_sort_column(text: str):
    text = text.lower()
    for phrase, col in SORTABLE_COLUMNS.items():
        if phrase in text:
            return col
    return None


def detect_ordering(text: str):
    text = text.lower()
    col = detect_sort_column(text)

    if "highest" in text or "largest" in text or "biggest" in text:
        return (col or "population", "DESC")

    if "lowest" in text or "smallest" in text:
        return (col or "population", "ASC")

    return (None, None)


 
# GROUP BY QUERIES
 

def detect_groupby(text: str):
    text = text.lower()
    if "by state" in text or "per state" in text or "group by state" in text:
        return "state"
    if "states" in text:
        return "state"
    return None


def detect_aggregation(text: str):
    text = text.lower()
    col = detect_sort_column(text)
    if not col:
        col = "population"

    if "total" in text or "sum" in text or "overall" in text:
        return ("SUM", col)
    if "avg" in text or "average" in text:
        return ("AVG", col)

    return ("AVG", col)


def build_groupby_sql(text: str):
    group_col = detect_groupby(text)
    agg_func, metric = detect_aggregation(text)
    limit = extract_limit(text)

    order_expr = f"{agg_func}({metric})"
    _, order_dir = detect_ordering(text)
    if not order_dir:
        order_dir = "DESC"

    if "top" in text or "first" in text:
        return f"""
        SELECT TOP ({limit}) {group_col}, {agg_func}({metric}) AS value
        FROM dbo.cities
        GROUP BY {group_col}
        ORDER BY {order_expr} {order_dir};
        """.strip()

    return f"""
    SELECT {group_col}, {agg_func}({metric}) AS value
    FROM dbo.cities
    GROUP BY {group_col}
    ORDER BY {order_expr} {order_dir};
    """.strip()


 
# PERCENTAGE QUERIES
 

def is_percentage_query(text: str):
    keywords = ["what percentage", "percentage of", "percent of", "what percent"]
    return any(k in text.lower() for k in keywords)


def build_percentage_sql(text: str):
    numeric = extract_numeric_filter(text)
    state = extract_state_filter(text)

    where = []
    if state: where.append(state)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    if numeric:
        cond = f"CASE WHEN {numeric} THEN 1 END"
    else:
        cond = "1"

    return f"""
    SELECT (100.0 * COUNT({cond}) / COUNT(*)) AS percentage
    FROM dbo.cities
    {where_sql};
    """.strip()


 
# TEXT SEARCH (LIKE)
 

def detect_text_search(text: str):
    text = text.lower()

    m = re.search(r"starting with ['\"]?([a-zA-Z ]+)['\"]?", text)
    if m:
        return ("starts_with", m.group(1).strip())

    m = re.search(r"ending with ['\"]?([a-zA-Z ]+)['\"]?", text)
    if m:
        return ("ends_with", m.group(1).strip())

    m = re.search(r"(containing|contains) ['\"]?([a-zA-Z ]+)['\"]?", text)
    if m:
        return ("contains", m.group(2).strip())

    return None


def build_text_search_sql(kind, value, limit):
    if kind == "starts_with":
        cond = f"city LIKE '{value}%'"
    elif kind == "ends_with":
        cond = f"city LIKE '%{value}'"
    else:
        cond = f"city LIKE '%{value}%'"

    return f"""
    SELECT TOP ({limit}) {", ".join(SELECT_COLUMNS)}
    FROM dbo.cities
    WHERE {cond}
    ORDER BY population DESC;
    """.strip()



 ----------------------------------
# BEST CITIES AI RANKING SYSTEM
 ----------------------------------

def is_best_query(text: str):
    text = text.lower()
    return ("best" in text) or ("top cities" in text)


def detect_best_profile(text: str):
    text = text.lower()

    if "family" in text:
        return "family_friendly"

    if "young" in text:
        return "young_people"

    if "senior" in text or "older" in text or "retire" in text:
        return "seniors"

    if "affordable" in text or "cheap" in text or "low cost" in text:
        return "affordable"

    return "general"


def get_best_scoring_formula(profile: str):
    if profile == "family_friendly":
        return """
            (0.5 * avg_household_size) +
            (0.3 * (1.0 / NULLIF(median_age,0))) +
            (0.2 * population)
        """

    if profile == "young_people":
        return """
            (0.4 * (1.0 / NULLIF(median_age,0))) +
            (0.3 * population) +
            (0.3 * avg_household_size)
        """

    if profile == "seniors":
        return """
            (0.6 * median_age) +
            (0.2 * population) +
            (0.2 * avg_household_size)
        """

    if profile == "affordable":
        return """
            (0.5 * (1.0 / NULLIF(avg_household_size,0))) +
            (0.3 * (1.0 / NULLIF(median_age,0))) +
            (0.2 * (1.0 / NULLIF(population,0)))
        """

    return """
        (0.4 * population) +
        (0.3 * avg_household_size) +
        (0.3 * (1.0 / NULLIF(median_age,0)))
    """


def build_best_sql(text: str):
    profile = detect_best_profile(text)
    formula = get_best_scoring_formula(profile)
    limit = extract_limit(text)
    state = extract_state_filter(text)

    where_sql = f"WHERE {state}" if state else ""

    return f"""
    SELECT TOP ({limit}) city, state, population, median_age, avg_household_size,
        {formula} AS score
    FROM dbo.cities
    {where_sql}
    ORDER BY score DESC;
    """.strip()



 
# MAIN SQL BUILDER
 

def build_sql(query: str):
    text = query.lower()

    # COUNT
    if is_count_query(text):
        return build_count_sql(text)

    # PERCENTAGE
    if is_percentage_query(text):
        return build_percentage_sql(text)

    # GROUP BY
    if detect_groupby(text):
        return build_groupby_sql(text)

    # BEST CITIES RANKING
    if is_best_query(text):
        return build_best_sql(text)

    # TEXT SEARCH
    text_search = detect_text_search(text)
    if text_search:
        kind, value = text_search
        limit = extract_limit(text)
        return build_text_search_sql(kind, value, limit)

    # NORMAL FILTERS
    limit = extract_limit(text)
    numeric = extract_numeric_filter(text)
    state = extract_state_filter(text)

    where = []
    if numeric: where.append(numeric)
    if state: where.append(state)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    order_col, order_dir = detect_ordering(text)
    if not order_col:
        order_col = "population"
        order_dir = "DESC"

    return f"""
    SELECT TOP ({limit}) {", ".join(SELECT_COLUMNS)}
    FROM dbo.cities
    {where_sql}
    ORDER BY {order_col} {order_dir};
    """.strip()
