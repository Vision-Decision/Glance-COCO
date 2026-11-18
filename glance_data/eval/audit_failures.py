"""Render a random sample of model failures (IoU<0.5) on an eval set: green = ground truth, red = prediction.

python -m glance_data.eval.audit_failures --rows out/model_rows/refcoco_val.jsonl --n 24 --out out/audit_refcoco_val
"""
import argparse
import json
import random

from PIL import Image, ImageDraw, ImageFont

from glance_data.common import IMG_ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--out", required=True)
ap.add_argument("--img-root", default=IMG_ROOT)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
rows = [json.loads(line) for line in open(a.rows)]
fail = [r for r in rows if r.get("iou") is not None and r["iou"] < 0.5]
print("n", len(rows), "fail", len(fail), "acc", 1 - len(fail) / len(rows))
random.seed(a.seed)
smp = random.sample(fail, min(a.n, len(fail)))
font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
tiles = []
for i, r in enumerate(smp):
    im = Image.open(f"{a.img_root}/{r['file']}").convert("RGB")
    s = 400 / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)))
    d = ImageDraw.Draw(im)
    d.rectangle([v * s for v in r["gt"]], outline=(0, 255, 0), width=3)
    d.rectangle([v * s for v in r["pred"]], outline=(255, 0, 0), width=3)
    t = Image.new("RGB", (400, 444), "white")
    t.paste(im, (0, 0))
    dd = ImageDraw.Draw(t)
    dd.text((4, 402), f"{i + 1}. {r['text']}"[:52], fill=(0, 0, 0), font=font)
    dd.text((4, 422), f"IoU {r['iou']:.2f}  P(present) {r['p_present']:.2f}", fill=(90, 90, 90), font=font)
    tiles.append(t)
for g in range(0, len(tiles), 12):
    sh = Image.new("RGB", (1600, 444 * 3), "gray")
    for k, t in enumerate(tiles[g:g + 12]):
        sh.paste(t, ((k % 4) * 400, (k // 4) * 444))
    sh.save(f"{a.out}_{g // 12 + 1}.jpg", quality=85)
json.dump(smp, open(f"{a.out}.json", "w"))
