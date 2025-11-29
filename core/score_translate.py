# core/score_translate.py

def to_level(score):
    """
    Convert numeric ML score to human-friendly category.
    Simple and understandable for end users.
    """

    if score is None:
        return "Unknown"

    try:
        score = float(score)
    except:
        return "Unknown"

    # Basic thresholds (you can tune later)
    if score < 5:
        return "Low"
    elif score < 15:
        return "Medium"
    elif score < 25:
        return "High"
    else:
        return "Excellent"
