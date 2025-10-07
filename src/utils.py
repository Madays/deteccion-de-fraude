import os, json
from typing import Dict
import joblib

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_model(obj, path: str):
    joblib.dump(obj, path)
