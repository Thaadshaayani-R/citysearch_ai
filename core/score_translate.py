# core/score_translate.py - Enhanced version

import numpy as np

def to_level(score):
    """Convert 0-100 score to label, emoji, and color."""
    if score is None:
        return "Unknown", "⚪", "#6b7280"
    try:
        score = float(score)
    except:
        return "Unknown", "⚪", "#6b7280"
    
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


def normalize_to_100(raw_score, all_scores=None):
    """
    Convert raw ML score to 0-100 using percentile ranking.
    """
    if raw_score is None:
        return 50.0
    
    raw_score = float(raw_score)
    
    if all_scores is not None and len(all_scores) > 0:
        all_scores = np.array(all_scores)
        percentile = (np.sum(all_scores < raw_score) / len(all_scores)) * 100
        return round(percentile, 1)
    else:
        # Fallback: map based on typical ranges
        if raw_score < 1:
            return round(raw_score * 100, 1)  # 0-1 range
        elif raw_score < 10:
            return round(raw_score * 10, 1)   # 0-10 range
        elif raw_score < 100:
            return round(raw_score, 1)        # Already 0-100
        else:
            return round(min(100, raw_score / 10), 1)  # Large numbers


def format_score_display(raw_score, all_scores=None):
    """
    Create a complete score display with label and emoji.
    """
    score_100 = normalize_to_100(raw_score, all_scores)
    label, emoji, color = to_level(score_100)
    
    return {
        "score_100": score_100,
        "score_raw": raw_score,
        "label": label,
        "emoji": emoji,
        "color": color,
    }
