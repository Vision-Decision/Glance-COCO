"""Dump per-question model predictions on eval sets (for auditing benchmark failures).

VLM_JEV_ROOT=~/experiments/jevbox_20261117/v2 python -m glance_data.eval.model_rows --ckpt CKPT --sets refcoco_val,refcocog_val --out out/model_rows
"""
import argparse
import json
import os

import torch

from glance_data.eval.vlm_jev_path import ensure

ensure()  # vlm-jev must be importable before the dv/ and eval imports below
from dv.config import Config  # noqa: E402
from dv.data.dataset import RecordDataset  # noqa: E402
from dv.model.build import build, load_ckpt  # noqa: E402
from eval import predict  # noqa: E402

from glance_data.common import PREP  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--sets", default="refcoco_val,refcocog_val")
ap.add_argument("--prep", default=PREP)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.backends.cuda.enable_cudnn_sdp(False)
sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = Config().update(sd["cfg"])
net, heads, proc = build(cfg, "cuda"); load_ckpt(a.ckpt, net, heads); net.eval(); heads.eval()
for name in a.sets.split(","):
    path = f"{a.prep}/eval_{name}.jsonl"
    recs = [json.loads(line) for line in open(path)]
    ds = RecordDataset(path, cfg.image_root, proc, cfg, train=False)
    rows = predict(net, heads, ds, images_per_batch=8)
    k = 0
    with open(f"{a.out}/{name}.jsonl", "w") as f:
        for r in recs:
            for q in r["questions"]:
                row = rows[k]; k += 1
                assert row["text"] == q["text"]
                f.write(json.dumps(dict(file=r["file"], image_id=r["image_id"], W=r["W"], H=r["H"], text=q["text"],
                                        present=q["present"], gt=q.get("box"), n_targets=q.get("n_targets", 1),
                                        boxes=q.get("boxes"), pred=row["box"], p_present=row["p_present"],
                                        p_iou50=row["p_iou50"], iou=row.get("iou"))) + "\n")
    print(name, k, flush=True)
