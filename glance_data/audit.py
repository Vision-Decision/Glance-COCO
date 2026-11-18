"""Stratified human audit of generated labels (data-side validation, no model involved).

1) sample: draw N queries per kind, render sheets (positives with their target boxes in green, negatives without
   boxes and marked "should be absent"), and write a judgement file to fill in.

   python -m glance_data.audit sample --aug out/aug_train.jsonl --inv out/inv_train --out out/audit --per-kind 8

2) review the sheets and set "correct": true/false for every item in out/audit/judgements.json

3) score: per-kind precision from the filled file.

   python -m glance_data.audit score --judgements out/audit/judgements.json
"""
import argparse
import collections
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

FONTS = ["/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/PingFang.ttc",
         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]


def font(size, path=None):
    for p in ([path] if path else []) + FONTS:
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def sample(a):
    recs = [json.loads(line) for line in open(a.aug)]
    by_kind = collections.defaultdict(list)
    for r in recs:
        for q in r["questions"]:
            by_kind[q["kind"]].append((r, q))
    rng = random.Random(a.seed)
    items = []
    for kind in sorted(by_kind):
        for r, q in rng.sample(by_kind[kind], min(a.per_kind, len(by_kind[kind]))):
            items.append(dict(id=len(items) + 1, kind=kind, lang=q.get("lang", "en"), image_id=r["image_id"],
                              text=q["text"], present=q["present"], n_targets=q["n_targets"], boxes=q["boxes"],
                              W=r["W"], H=r["H"], correct=None, note=""))
    os.makedirs(a.out, exist_ok=True)
    TW, TH = 420, 330
    f13, f14 = font(13, a.font), font(14, a.font)
    for s in range(0, len(items), 8):
        sheet = Image.new("RGB", (TW * 4, TH * 2), (200, 200, 200))
        for j, it in enumerate(items[s:s + 8]):
            im = Image.open(f"{a.inv}/img/{it['image_id']}_clean.jpg").convert("RGB")
            sc = im.width / it["W"]
            d = ImageDraw.Draw(im)
            for b in it["boxes"] or []:
                d.rectangle([v * sc for v in b], outline=(0, 255, 0), width=3)
            im.thumbnail((TW, TH - 58))
            t = Image.new("RGB", (TW, TH), "white")
            t.paste(im, (0, 0))
            dd = ImageDraw.Draw(t)
            tag = f"{it['id']}. [{it['kind']}] " + (f"present n={it['n_targets']}" if it["present"] else "should be absent")
            dd.text((4, TH - 56), tag, fill=(0, 100, 0) if it["present"] else (140, 0, 0), font=f13)
            dd.text((4, TH - 38), it["text"][:58], fill=(0, 0, 0), font=f14)
            dd.text((4, TH - 20), it["text"][58:116], fill=(0, 0, 0), font=f14)
            sheet.paste(t, ((j % 4) * TW, (j // 4) * TH))
        sheet.save(f"{a.out}/sheet_{s // 8 + 1:02d}.jpg", quality=85)
    json.dump(items, open(f"{a.out}/judgements.json", "w"), ensure_ascii=False, indent=1)
    print(f"{len(items)} items, {len(by_kind)} kinds -> {a.out}/sheet_*.jpg and {a.out}/judgements.json")


def score(a):
    items = json.load(open(a.judgements))
    per = collections.defaultdict(lambda: [0, 0])
    todo = 0
    for it in items:
        if it.get("correct") is None:
            todo += 1
            continue
        per[it["kind"]][0] += bool(it["correct"])
        per[it["kind"]][1] += 1
    res = {k: dict(correct=c, n=n, precision=c / n) for k, (c, n) in sorted(per.items())}
    tot_c, tot_n = sum(v["correct"] for v in res.values()), sum(v["n"] for v in res.values())
    for k, v in res.items():
        print(f"{k:16s} {v['correct']}/{v['n']}")
    print(f"total {tot_c}/{tot_n} = {tot_c / max(1, tot_n):.1%}" + (f"  ({todo} not judged yet)" if todo else ""))
    if a.out:
        json.dump(dict(per_kind=res, total=dict(correct=tot_c, n=tot_n), not_judged=todo), open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--aug", required=True)
    s.add_argument("--inv", required=True, help="inventory folder of the same images (for the 640px images)")
    s.add_argument("--out", required=True)
    s.add_argument("--per-kind", type=int, default=8)
    s.add_argument("--seed", type=int, default=5)
    s.add_argument("--font", default=None, help="a TTF/TTC with CJK glyphs, if the defaults are missing")
    c = sub.add_parser("score")
    c.add_argument("--judgements", required=True)
    c.add_argument("--out", default="")
    a = ap.parse_args()
    sample(a) if a.cmd == "sample" else score(a)


if __name__ == "__main__":
    main()
