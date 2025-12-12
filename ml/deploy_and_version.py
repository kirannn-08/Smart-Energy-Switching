# ml/deploy_and_version.py
"""
Deploy & version helper.

Usage (from project root):
    python3 -c "import ml.deploy_and_version as dv; dv.deploy_and_register('ml/pv_lstm_20250101.pt','ml/load_lstm_20250101.pt','retrain comment')"
Or import and call `deploy_and_register(pv_path, load_path, comment=...)`
"""

import os
import json
import shutil
from datetime import datetime

MODELS_JSON = "ml/models.json"
PV_LATEST = "ml/pv_lstm_latest.pt"
LOAD_LATEST = "ml/load_lstm_latest.pt"

def deploy_and_register(pv_model_path, load_model_path, comment=""):
    # verify paths
    for p in (pv_model_path, load_model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Model path not found: {p}")

    # copy to timestamped files inside ml/ if they are outside
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pv_target = os.path.join("ml", f"pv_lstm_{timestamp}.pt")
    load_target = os.path.join("ml", f"load_lstm_{timestamp}.pt")

    shutil.copyfile(pv_model_path, pv_target)
    shutil.copyfile(load_model_path, load_target)

    # update latest symlink / file
    for src, dst in ((pv_target, PV_LATEST), (load_target, LOAD_LATEST)):
        try:
            if os.path.islink(dst) or os.path.exists(dst):
                try:
                    os.remove(dst)
                except Exception:
                    os.remove(dst)
            os.symlink(os.path.basename(src), dst)
        except Exception:
            # fallback: copy
            shutil.copyfile(src, dst)

    # update models.json
    meta = {}
    if os.path.exists(MODELS_JSON):
        with open(MODELS_JSON, "r") as f:
            try:
                meta = json.load(f)
            except Exception:
                meta = {"models": []}
    else:
        meta = {"models": []}

    entry = {
        "timestamp": datetime.now().isoformat(),
        "pv_model": pv_target,
        "load_model": load_target,
        "comment": comment,
    }
    meta["models"].append(entry)
    with open(MODELS_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    print("Deployed models:")
    print("  PV ->", pv_target)
    print("  Load ->", load_target)
    print("Updated latest pointers and ml/models.json")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        pv, load = sys.argv[1], sys.argv[2]
        comment = sys.argv[3] if len(sys.argv) > 3 else ""
        deploy_and_register(pv, load, comment)
    else:
        print("Example usage:")
        print(" python3 ml/deploy_and_version.py ml/pv_lstm_20250101.pt ml/load_lstm_20250101.pt 'weekly retrain'")