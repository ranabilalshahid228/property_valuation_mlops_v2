import json, os, time, joblib
from typing import Dict, List

REGISTRY_PATH = "models/registry.json"
ACTIVE_LINK = "models/active_model.joblib"

def _now_id():
    return time.strftime("%Y%m%d_%H%M%S")

def _load() -> Dict:
    if not os.path.exists(REGISTRY_PATH):
        return {"versions": []}
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def _save(reg: Dict):
    os.makedirs("models", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)

def register_model(name: str, path: str, metrics: Dict):
    reg = _load()
    version = _now_id()
    target = f"models/model_{version}.joblib"
    if os.path.exists(path):
        import shutil
        shutil.move(path, target)
    entry = {"version": version, "name": name, "path": target, "metrics": metrics, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    reg["versions"].insert(0, entry)
    _save(reg)
    activate_version(version)
    return version

def list_versions() -> List[Dict]:
    return _load().get("versions", [])

def activate_version(version: str):
    reg = _load()
    match = next((v for v in reg.get("versions", []) if v["version"] == version), None)
    if not match:
        raise ValueError("Version not found")
    import shutil
    shutil.copyfile(match["path"], ACTIVE_LINK)
    reg["active"] = version
    _save(reg)
    return match

def active_version() -> str:
    return _load().get("active","")
