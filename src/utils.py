import os, json, joblib

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_model(model, path):
    joblib.dump(model, path)
