# core/score_translate.py

import numpy as np

def to_level(score):
    """Convert 0-100 score to label."""
    if score is None:
        return "Unknown", "⚪", "#gray"
    try:
        score = float(score)
    except:
        return "Unknown", "⚪", "#gray"
    
    if score >= 90:
        return "Excellent Match", "⭐", "#FFD700"
    elif score >= 75:
        return "Great Match", "🟢", "#22c55e"
    elif score >= 60:
        return "Good Match", "🔵", "#3b82f6"
    elif score >= 40:
        return "Average Match", "🟡", "#eab308"
    elif score >= 20:
        return "Below Average", "🟠", "#f97316"
    else:
        return "Low Match", "🔴", "#ef4444"


def normalize_to_100(raw_score, all_scores):
    """
    Convert raw ML score to 0-100 using percentile ranking.
    
    Args:
        raw_score: The city's raw score
        all_scores: Array/list of all scores in the dataset
    
    Returns:
        Float between 0-100
    """
    if all_scores is None or len(all_scores) == 0:
        return 50.0
    
    all_scores = np.array(all_scores)
    percentile = (np.sum(all_scores < raw_score) / len(all_scores)) * 100
    return round(percentile, 1)


def format_score_display(raw_score, all_scores=None):
    """
    Create a complete score display with label and emoji.
    
    Returns dict with all display info.
    """
    # Normalize to 0-100
    if all_scores is not None:
        score_100 = normalize_to_100(raw_score, all_scores)
    else:
        # Fallback: assume raw score is already reasonable
        # Map typical ranges to 0-100
        if raw_score < 1:
            score_100 = raw_score * 100  # 0-1 range
        elif raw_score < 10:
            score_100 = raw_score * 10   # 0-10 range
        elif raw_score < 100:
            score_100 = raw_score        # Already 0-100
        else:
            score_100 = min(100, raw_score / 10)  # Large numbers
    
    label, emoji, color = to_level(score_100)
    
    return {
        "score_100": score_100,
        "score_raw": raw_score,
        "label": label,
        "emoji": emoji,
        "color": color,
        "display_text": f"{score_100:.0f}/100 {emoji} {label}",
        "bar_percent": score_100
    }

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
