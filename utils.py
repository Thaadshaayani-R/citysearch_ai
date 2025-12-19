# utils.py
import re
import pandas as pd
from difflib import get_close_matches
from config import US_STATES_FULL, US_STATES_LOWER, PROTECTED_WORDS, FORBIDDEN_WORLD_KEYWORDS


 
# MARKDOWN TO HTML CONVERTER
 
def convert_markdown_to_html(text: str) -> str:
    """Convert common Markdown formatting to HTML."""
    
    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert *italic* to <em>italic</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Convert bullet points (lines starting with - or •)
    lines = text.split('\n')
    converted_lines = []
    in_list = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- ') or stripped.startswith('• '):
            if not in_list:
                converted_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_list = True
            item_text = stripped[2:].strip()
            converted_lines.append(f'<li style="margin: 0.25rem 0;">{item_text}</li>')
        elif stripped.startswith('* ') and not stripped.startswith('**'):
            if not in_list:
                converted_lines.append('<ul style="margin: 0.5rem 0; padding-left: 1.5rem;">')
                in_list = True
            item_text = stripped[2:].strip()
            converted_lines.append(f'<li style="margin: 0.25rem 0;">{item_text}</li>')
        else:
            if in_list:
                converted_lines.append('</ul>')
                in_list = False
            if stripped:
                converted_lines.append(f'<p style="margin: 0.5rem 0;">{stripped}</p>')
    
    if in_list:
        converted_lines.append('</ul>')
    
    return '\n'.join(converted_lines)


 
# CSV CONVERSION
 
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')


 
# SAFETY FILTERS
 
def is_nonsense_query(q: str) -> bool:
    """Check if query is nonsensical."""
    if len(q.strip()) < 2:
        return True
    if re.fullmatch(r"[0-9]+", q.strip()):
        return True
    return False


def is_world_query(q: str) -> bool:
    """Check if query is about non-US locations."""
    q_lower = q.lower()
    return any(keyword in q_lower for keyword in FORBIDDEN_WORLD_KEYWORDS)


def is_city_related_query(q: str) -> bool:
    """Check if query is related to cities/states."""
    city_keywords = [
        "city", "cities", "state", "states", "population", "cluster", "compare",
        "texas", "florida", "california", "new york", "illinois", "ohio",
        "best", "similar", "lifestyle", "profile", "retirement", "family",
        "professionals", "median", "age", "household", "denver", "miami",
        "dallas", "austin", "chicago", "phoenix", "seattle", "boston",
        "atlanta", "houston", "los angeles", "san francisco"
    ]
    
    q_lower = q.lower()
    return any(keyword in q_lower for keyword in city_keywords)


 
# FUZZY MATCHING - STATES
 
def fuzzy_match_state(input_state: str, cutoff: float = 0.6) -> str:
    """Find the closest matching state name."""
    if not input_state:
        return None
    
    input_lower = input_state.lower().strip()
    
    # Exact match first
    for state in US_STATES_FULL:
        if state.lower() == input_lower:
            return state
    
    # Fuzzy match
    matches = get_close_matches(input_lower, US_STATES_LOWER, n=1, cutoff=cutoff)
    
    if matches:
        matched_index = US_STATES_LOWER.index(matches[0])
        return US_STATES_FULL[matched_index]
    
    return None


def extract_single_state_fuzzy(q: str) -> str:
    """Extract a single state name from query with fuzzy matching."""
    q_lower = q.lower()
    
    # Exact match first
    for state in US_STATES_FULL:
        if state.lower() in q_lower:
            return state
    
    # Fuzzy match
    words = q_lower.replace(",", " ").replace(".", " ").replace("?", " ").split()
    
    # Two-word combinations first (New York, North Carolina, etc.)
    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        matched = fuzzy_match_state(two_word, cutoff=0.7)
        if matched:
            return matched
    
    # Single words
    for word in words:
        if len(word) > 3:
            matched = fuzzy_match_state(word, cutoff=0.7)
            if matched:
                return matched
    
    return None


def extract_two_states_fuzzy(q: str) -> list:
    """Extract two state names from query with fuzzy matching."""
    q_lower = q.lower()
    found = []
    
    # First pass: exact matches
    for state in US_STATES_FULL:
        if state.lower() in q_lower:
            found.append(state)
    
    if len(set(found)) == 2:
        return list(set(found))[:2]
    
    # Second pass: fuzzy matching
    words = q_lower.replace(",", " ").replace(".", " ").replace("?", " ").split()
    
    # Try two-word combinations first
    for i in range(len(words) - 1):
        two_word = f"{words[i]} {words[i+1]}"
        matched = fuzzy_match_state(two_word, cutoff=0.7)
        if matched and matched not in found:
            found.append(matched)
    
    # Then single words
    for word in words:
        if len(word) > 3:
            matched = fuzzy_match_state(word, cutoff=0.7)
            if matched and matched not in found:
                found.append(matched)
    
    found = list(set(found))
    if len(found) >= 2:
        return found[:2]
    
    return None


 
# FUZZY MATCHING - CITIES
 
def fuzzy_match_city(input_city: str, city_list: list, cutoff: float = 0.6) -> str:
    """Find the closest matching city name from a list."""
    if not input_city:
        return None
    
    input_lower = input_city.lower().strip()
    city_names_lower = [c.lower() for c in city_list]
    
    # Exact match first
    if input_lower in city_names_lower:
        matched_index = city_names_lower.index(input_lower)
        return city_list[matched_index]
    
    # Fuzzy match
    matches = get_close_matches(input_lower, city_names_lower, n=1, cutoff=cutoff)
    
    if matches:
        matched_index = city_names_lower.index(matches[0])
        return city_list[matched_index]
    
    return None


def extract_single_city_fuzzy(q: str, city_list: list) -> str:
    """Extract a single city name from query with fuzzy matching."""
    q_lower = q.lower()
    
    # Common query words to skip
    QUERY_SKIP_WORDS = {
        "cities", "city", "population", "median", "average", "household", 
        "size", "age", "total", "count", "how", "many", "what", "which",
        "best", "worst", "top", "bottom", "largest", "smallest", "biggest",
        "highest", "lowest", "greater", "less", "more", "than", "over", 
        "under", "above", "below", "with", "from", "the", "and", "for", 
        "are", "this", "that", "number", "percent", "percentage"
    }
    
    # Get state names dynamically
    state_names_lower = {s.lower() for s in US_STATES_FULL}
    
    # Combine skip words
    all_skip_words = QUERY_SKIP_WORDS | state_names_lower
    
    city_names_lower = [c.lower() for c in city_list]
    
    # Exact match first - only if the city name appears as a complete word
    for i, city_lower in enumerate(city_names_lower):
        # Skip if city name is also a state name (e.g., "New York")
        if city_lower in state_names_lower:
            continue
            
        # Check for exact word match using word boundaries
        pattern = r'\b' + re.escape(city_lower) + r'\b'
        if re.search(pattern, q_lower):
            return city_list[i]
    
    # Fuzzy match - but be very careful
    words = q_lower.replace(",", " ").replace(".", " ").replace("?", " ").split()
    
    for word in words:
        # Skip short words
        if len(word) <= 3:
            continue
            
        # Skip if this word is in our skip list
        if word in all_skip_words:
            continue
        
        # Only do fuzzy matching with high cutoff (0.85 = very similar)
        matched = fuzzy_match_city(word, city_list, cutoff=0.85)
        if matched and matched.lower() not in all_skip_words:
            return matched
    
    return None


def extract_two_cities_fuzzy(q: str, city_list: list) -> list:
    """Extract two city names from query with fuzzy matching."""
    q_lower = q.lower()
    city_names_lower = [c.lower() for c in city_list]
    
    found_original = []
    
    # First pass: exact matches
    for i, city_lower in enumerate(city_names_lower):
        if city_lower in q_lower and city_list[i] not in found_original:
            found_original.append(city_list[i])
    
    if len(found_original) == 2:
        return found_original
    
    # Second pass: fuzzy matching
    words = q_lower.replace(",", " ").replace(".", " ").replace("?", " ").split()
    
    for word in words:
        if len(word) > 3:
            matched = fuzzy_match_city(word, city_list, cutoff=0.7)
            if matched and matched not in found_original:
                found_original.append(matched)
    
    if len(found_original) >= 2:
        return found_original[:2]
    
    return None


 
# SPELLING CORRECTION
 
def correct_query_spelling(query: str, city_list: list) -> tuple:
    """
    Correct spelling mistakes in the query for states and cities.
    Returns: (corrected_query, list_of_corrections)
    
    IMPORTANT: Only corrects words that look like misspelled city/state names,
    NOT common English words.
    """
    
    corrections = []
    corrected_query = query
    words = query.split()
    
    city_names_lower = [c.lower() for c in city_list]
    state_names_lower = [s.lower() for s in US_STATES_FULL]
    
    i = 0
    while i < len(words):
        word = words[i]
        word_clean = word.lower().strip(".,?!;:'\"")
        
        # Skip short words
        if len(word_clean) <= 3:
            i += 1
            continue
        
        # Skip protected words
        if word_clean in PROTECTED_WORDS:
            i += 1
            continue
        
        # Skip if it's already a valid city or state (exact match)
        if word_clean in city_names_lower or word_clean in state_names_lower:
            i += 1
            continue
        
        # Check two-word combinations first (New York, North Carolina, etc.)
        if i < len(words) - 1:
            next_word = words[i+1].lower().strip(".,?!;:'\"")
            two_word = f"{word_clean} {next_word}"
            
            # Skip if either word is protected
            if word_clean in PROTECTED_WORDS or next_word in PROTECTED_WORDS:
                pass  # Don't try two-word match
            else:
                matched_state = fuzzy_match_state(two_word, cutoff=0.85)
                if matched_state and matched_state.lower() != two_word:
                    original = f"{word} {words[i+1]}"
                    corrections.append((original.strip(".,?!;:'\""), matched_state))
                    corrected_query = corrected_query.replace(original.strip(".,?!;:'\""), matched_state, 1)
                    i += 2
                    continue
        
        # Only try fuzzy matching if the word LOOKS like a city/state name
        should_try_fuzzy = (
            word[0].isupper() or  # Capitalized
            len(word_clean) >= 6   # Long enough to be a place name
        )
        
        if not should_try_fuzzy:
            i += 1
            continue
        
        # Try single word - state match (high cutoff)
        matched_state = fuzzy_match_state(word_clean, cutoff=0.8)
        if matched_state and matched_state.lower() != word_clean:
            corrections.append((word_clean, matched_state))
            corrected_query = corrected_query.replace(word, matched_state, 1)
            i += 1
            continue
        
        # Try single word - city match (high cutoff)
        matched_city = fuzzy_match_city(word_clean, city_list, cutoff=0.8)
        if matched_city and matched_city.lower() != word_clean:
            corrections.append((word_clean, matched_city))
            corrected_query = corrected_query.replace(word, matched_city, 1)
            i += 1
            continue
        
        i += 1
    
    return corrected_query, corrections


 
# COMPARISON HELPERS
 
def is_comparison_query(q: str) -> bool:
    """Check if query is a comparison query."""
    q_lower = q.lower()
    comparison_words = ["vs", "versus", "compare", "comparison", "between", "or", "better"]
    return any(word in q_lower for word in comparison_words)


def is_state_comparison_query(q: str) -> bool:
    """Check if query is comparing states."""
    if not is_comparison_query(q):
        return False
    
    states_found = extract_two_states_fuzzy(q)
    return states_found is not None


def is_city_comparison_query(q: str, city_list: list) -> bool:
    """Check if query is comparing cities."""
    if not is_comparison_query(q):
        return False
    
    cities_found = extract_two_cities_fuzzy(q, city_list)
    return cities_found is not None


 
# FORMAT HELPERS
 
def format_number(value) -> str:
    """Format a number with commas."""
    if isinstance(value, (int, float)):
        if value > 1000:
            return f"{int(value):,}"
        elif isinstance(value, float):
            return f"{value:.2f}"
        else:
            return str(value)
    return str(value)


def format_population(value) -> str:
    """Format population number."""
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    return str(value)


def format_age(value) -> str:
    """Format age value."""
    if isinstance(value, (int, float)):
        return f"{value:.1f} years"
    return str(value)


def format_household_size(value) -> str:
    """Format household size value."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)
