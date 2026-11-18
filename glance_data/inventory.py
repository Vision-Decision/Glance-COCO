"""Stage 0 (server): per-image object inventory + marked images for the generator.

For each image: all COCO instances (exact boxes), the human expressions already attached to each instance
(RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO single-target, of the allowed splits), gRefCOCO multi-target and
no-target sentences, and up to K "marked" objects that the LLM will write about. Marked objects are drawn as
numbered boxes on a copy of the image (set-of-mark); the clean image is sent too.

python -m glance_data.inventory --images train --out out/inv_train
python -m glance_data.inventory --images refcoco_val --out out/inv_val --limit 300
"""
import argparse
import collections
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

from glance_data.common import IMG_ROOT, PREP, load_grefs, load_instances, load_refs

COLORS = [(255, 40, 40), (40, 200, 40), (40, 120, 255), (255, 160, 0), (220, 40, 220), (0, 200, 200), (250, 250, 50),
          (255, 255, 255)]


def pick_marked(objs, expr, rng, K=6):
    """objs: instances (with 'id','cat','area'); expr: ann_id -> [sentences]. Prefer human-described objects, then
    undescribed siblings of described categories (coverage gap), then other undescribed objects (largest first)."""
    described = sorted([o for o in objs if expr.get(o["id"])], key=lambda o: -len(expr[o["id"]]))
    rng.shuffle(described)
    first = []
    cats_first = collections.Counter()
    for o in described:                                    # up to 4, at most 3 per category
        if len(first) < 4 and cats_first[o["cat"]] < 3:
            first.append(o); cats_first[o["cat"]] += 1
    used = {o["id"] for o in first}
    sib = [o for o in objs if o["id"] not in used and o["cat"] in cats_first]
    sib.sort(key=lambda o: -o["area"])
    second = sib[:2]
    used |= {o["id"] for o in second}
    rest = sorted([o for o in objs if o["id"] not in used], key=lambda o: -o["area"])
    return (first + second + rest)[:K]


def draw_marks(img, marked):
    im = img.copy()
    d = ImageDraw.Draw(im)
    s = max(im.size) / 640
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(18 * s))
    except OSError:
        font = ImageFont.load_default()
    for k, o in enumerate(marked):
        c = COLORS[k % len(COLORS)]
        d.rectangle(o["box"], outline=c, width=max(2, int(3 * s)))
        x, y = o["box"][0], o["box"][1]
        tag = str(k + 1)
        tw = d.textlength(tag, font=font)
        d.rectangle([x, y, x + tw + 8 * s, y + 22 * s], fill=c)
        d.text((x + 4 * s, y), tag, fill=(0, 0, 0), font=font)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="train", help="train | refcoco_val | refcocog_val")
    ap.add_argument("--prep", default=PREP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-area", type=float, default=0.004, help="min instance area / image area to be markable")
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(f"{a.out}/img", exist_ok=True)
    images, anns, cats = load_instances()
    refs = load_refs()
    grefs = load_grefs()
    if a.images == "train":
        img_ids = sorted({json.loads(line)["image_id"] for line in open(f"{a.prep}/train.jsonl")})
        ok_split = {"train"}
    else:
        img_ids = sorted({json.loads(line)["image_id"] for line in open(f"{a.prep}/eval_{a.images}.jsonl")})
        ok_split = {"val", "testA", "testB", "test"}
    rng.shuffle(img_ids)
    if a.limit:
        img_ids = img_ids[: a.limit]
    want = set(img_ids)
    expr = collections.defaultdict(list)                      # ann_id -> sentences (allowed splits only)
    for r in refs:
        if r["image_id"] in want and r["split"] in ok_split:
            expr[r["ann_id"]] += [s for s in r["sents"] if s not in expr[r["ann_id"]]]
    gmulti, gnone = collections.defaultdict(list), collections.defaultdict(list)
    for g in grefs:
        if g["image_id"] not in want or g["split"] not in ({"train"} if a.images == "train" else {"val", "testA", "testB"}):
            continue
        ids = [x for x in g["ann_id"] if x != -1]
        for s in g["sentences"]:
            if g["no_target"]:
                gnone[g["image_id"]].append(s["sent"])
            elif len(ids) > 1:
                gmulti[g["image_id"]].append(dict(text=s["sent"], ann_ids=ids))
            elif s["sent"] not in expr[ids[0]]:
                expr[ids[0]].append(s["sent"])
    n = 0
    with open(f"{a.out}/inventory.jsonl", "w") as f:
        for iid in img_ids:
            W, H = images[iid]["width"], images[iid]["height"]
            allo = anns[iid]
            objs = [o for o in allo if not o["crowd"] and o["area"] >= a.min_area * W * H]
            if not objs:
                continue
            marked = pick_marked(objs, expr, rng, a.K)
            file = f"COCO_train2014_{iid:012d}.jpg"
            img = Image.open(f"{IMG_ROOT}/{file}").convert("RGB")
            sc = 640 / max(W, H)
            small = img.resize((round(W * sc), round(H * sc)), Image.BICUBIC)
            ms = [dict(o, box=[v * sc for v in o["box"]]) for o in marked]
            small.save(f"{a.out}/img/{iid}_clean.jpg", quality=88)
            draw_marks(small, ms).save(f"{a.out}/img/{iid}_marked.jpg", quality=88)
            cnt = collections.Counter(cats[o["cat"]] for o in allo if not o["crowd"])
            crowd = sorted({cats[o["cat"]] for o in allo if o["crowd"]})
            rec = dict(image_id=iid, file=file, W=W, H=H, counts=dict(cnt), crowd_cats=crowd,
                       marked=[dict(mark=k + 1, ann_id=o["id"], cat=cats[o["cat"]], box=o["box"],
                                    box_norm=[round(o["box"][0] / W, 3), round(o["box"][1] / H, 3),
                                              round(o["box"][2] / W, 3), round(o["box"][3] / H, 3)],
                                    human=expr.get(o["id"], [])[:5]) for k, o in enumerate(marked)],
                       instances=[dict(ann_id=o["id"], cat=cats[o["cat"]], box=o["box"], crowd=o["crowd"]) for o in allo],
                       gref_multi=gmulti.get(iid, []), gref_none=gnone.get(iid, []))
            f.write(json.dumps(rec) + "\n")
            n += 1
    print("images", n, "->", a.out)


if __name__ == "__main__":
    main()
