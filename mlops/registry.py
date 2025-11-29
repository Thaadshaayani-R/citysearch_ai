import json
import os

# get the directory of THIS file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# correct path → same folder as this file
REGISTRY_FILE = os.path.join(BASE_DIR, "registry.json")

def load_registry():
    """Load the model registry metadata safely."""
    if not os.path.exists(REGISTRY_FILE):
        raise FileNotFoundError(f"Registry file not found at: {REGISTRY_FILE}")

    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)


def save_registry(data: dict):
    """Save updated registry file."""
    with open(REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=4)
