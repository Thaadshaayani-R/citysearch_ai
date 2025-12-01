# mlops/registry.py (FIXED VERSION)
"""
Unified model registry for CitySearch AI MLOps.
Single source of truth for model metadata.

Usage:
    from mlops.registry import (
        load_registry,
        save_registry,
        update_registry,
        get_model_path,
    )
"""

import os
import json
from datetime import datetime
from typing import Optional


# ===========================================
# CONFIGURATION
# ===========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_FILE = os.path.join(BASE_DIR, "registry.json")
ARCHIVE_DIR = os.path.join(BASE_DIR, "registry", "archive")


# ===========================================
# DEFAULT REGISTRY SCHEMA
# ===========================================
DEFAULT_REGISTRY = {
    "model_name": "city_kmeans",
    "version": "1.0",
    "model_path": "models/city_clusters.pkl",
    "n_clusters": 5,
    "silhouette_score": 0.0,
    "trained_on": None,
    "num_cities": 0,
    "drift_detected": False,
    "feature_columns": [
        "ml_vector_population",
        "ml_vector_age",
        "ml_vector_household"
    ],
    "notes": ""
}


# ===========================================
# CORE FUNCTIONS
# ===========================================
def load_registry() -> dict:
    """
    Load the model registry metadata.
    
    Returns:
        dict: Registry data
    
    Raises:
        FileNotFoundError: If registry file doesn't exist
    """
    if not os.path.exists(REGISTRY_FILE):
        raise FileNotFoundError(
            f"Registry file not found at: {REGISTRY_FILE}\n"
            "Run training first to create the registry."
        )
    
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(data: dict) -> None:
    """
    Save registry data to file.
    
    Args:
        data: Registry data to save
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def update_registry(**kwargs) -> dict:
    """
    Update specific fields in the registry.
    
    Args:
        **kwargs: Fields to update
    
    Returns:
        Updated registry data
    """
    try:
        registry = load_registry()
    except FileNotFoundError:
        registry = DEFAULT_REGISTRY.copy()
    
    registry.update(kwargs)
    save_registry(registry)
    return registry


def init_registry() -> dict:
    """
    Initialize a new registry with default values.
    
    Returns:
        New registry data
    """
    registry = DEFAULT_REGISTRY.copy()
    registry["trained_on"] = datetime.utcnow().isoformat()
    save_registry(registry)
    return registry


# ===========================================
# HELPER FUNCTIONS
# ===========================================
def get_model_path() -> str:
    """Get the current model path from registry."""
    try:
        registry = load_registry()
        return registry.get("model_path", "models/city_clusters.pkl")
    except FileNotFoundError:
        return "models/city_clusters.pkl"


def get_model_version() -> str:
    """Get the current model version from registry."""
    try:
        registry = load_registry()
        return registry.get("version", "1.0")
    except FileNotFoundError:
        return "1.0"


def increment_version(current_version: str) -> str:
    """
    Increment version number.
    
    Args:
        current_version: Current version string (e.g., "1.0")
    
    Returns:
        New version string (e.g., "1.1")
    """
    try:
        major, minor = current_version.split(".")
        new_minor = int(minor) + 1
        return f"{major}.{new_minor}"
    except (ValueError, AttributeError):
        return "1.1"


def registry_exists() -> bool:
    """Check if registry file exists."""
    return os.path.exists(REGISTRY_FILE)


def get_registry_safe() -> Optional[dict]:
    """
    Load registry without raising errors.
    
    Returns:
        Registry data or None if not found
    """
    try:
        return load_registry()
    except FileNotFoundError:
        return None


# ===========================================
# ARCHIVE FUNCTIONS
# ===========================================
def archive_current_model() -> Optional[str]:
    """
    Archive the current model before retraining.
    
    Returns:
        Archive path or None if no model to archive
    """
    import shutil
    
    try:
        registry = load_registry()
    except FileNotFoundError:
        return None
    
    current_path = registry.get("model_path")
    version = registry.get("version", "unknown")
    
    if not current_path or not os.path.exists(current_path):
        return None
    
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    filename = os.path.basename(current_path)
    name, ext = os.path.splitext(filename)
    archive_name = f"{name}_v{version}{ext}"
    archive_path = os.path.join(ARCHIVE_DIR, archive_name)
    
    shutil.copy(current_path, archive_path)
    return archive_path
