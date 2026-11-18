"""Make the vlm-jev repository importable (dv/, eval.py, train.py)."""
import os
import sys


def ensure():
    root = os.path.expanduser(os.environ.get("VLM_JEV_ROOT", "~/experiments/jevbox_20261117/v2"))
    if not os.path.isdir(os.path.join(root, "dv")):
        raise SystemExit(f"VLM_JEV_ROOT={root} is not a vlm-jev checkout (no dv/ folder)")
    if root not in sys.path:
        sys.path.insert(0, root)
