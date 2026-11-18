"""Evaluate a box checkpoint on an augmented record file (e.g. the val diagnostic set), broken down by query kind
and language.

VLM_JEV_ROOT=~/experiments/jevbox_20261117/v2 python -m glance_data.eval.diagnose --ckpt CKPT --records out/aug_val.jsonl --out out/diag.json

Metrics per kind:
  single-target positives : acc50 (IoU>=0.5, ignoring presence), present_rate (P(present)>=0.5)
  multi-target positives  : present_rate, hit_any (predicted box IoU>=0.5 with any target; the head has one box)
  negatives               : reject_rate (P(present)<0.5)
"""
import argparse
import collections
import json

import torch

from glance_data.eval.vlm_jev_path import ensure

ensure()  # vlm-jev must be importable before the dv/ and eval imports below
from dv.config import Config  # noqa: E402
from dv.data.dataset import RecordDataset  # noqa: E402
from dv.model.build import build, load_ckpt  # noqa: E402
from dv.model.heads import box_iou_giou  # noqa: E402
from eval import predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.backends.cuda.enable_cudnn_sdp(False)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = Config().update(sd["cfg"])
    net, heads, proc = build(cfg, "cuda")
    load_ckpt(a.ckpt, net, heads)
    net.eval(); heads.eval()
    recs = [json.loads(line) for line in open(a.records)]
    ds = RecordDataset(a.records, cfg.image_root, proc, cfg, train=False)
    rows = predict(net, heads, ds, images_per_batch=4)
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    k = 0
    for r in recs:
        for q in r["questions"]:
            row = rows[k]; k += 1
            p = row["p_present"]
            for key in (q["kind"], f"{q['kind']}:{q.get('lang', 'en')}", "ALL_POS" if q["present"] else "ALL_NEG"):
                m = agg[key]
                m["n"] += 1
                if not q["present"]:
                    m["reject_rate"] += p < 0.5
                    continue
                m["present_rate"] += p >= 0.5
                pred = torch.tensor(row["box"])[None]
                ious = [float(box_iou_giou(pred, torch.tensor(b, dtype=torch.float)[None])[0][0]) for b in q["boxes"]]
                if q["n_targets"] == 1:
                    m["acc50"] += ious[0] >= 0.5
                else:
                    m["hit_any"] += max(ious) >= 0.5
    assert k == len(rows)
    res = {key: {mk: (v / m["n"] if mk != "n" else int(v)) for mk, v in m.items()} for key, m in sorted(agg.items())}
    json.dump(res, open(a.out, "w"), indent=1)
    for key, m in res.items():
        print(f"{key:28s} " + " ".join(f"{mk} {v:.3f}" if mk != "n" else f"n {v}" for mk, v in m.items()))


if __name__ == "__main__":
    main()
