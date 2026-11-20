"""Minimal pairs: give rule-made questions a textually identical twin of the other polarity on another image.

A rule template writes its answer into the text: "two clocks" exists because the image does NOT have two clocks, and
"the cup on the left" because the only cup is on the right. A model can then learn presence from the string alone.
Here every such string also appears with the opposite answer on a different image, so the string carries no signal
and only the image can decide. Ground truth for every new question comes from COCO annotations:

  count     neg_count "two clocks" / "两个钟" -> the same string, PRESENT, on an image with exactly that many clocks
            (count_min <= n <= count_max, every instance reliable, no pile-prone category). A negative whose string has
            no such image is DROPPED, so no count string is left as a lone negative.
  position  neg_position "the cup on the left" / "左边的杯子" -> the same string, PRESENT, on an image whose only cup is
            reliable and clearly on that side (centre in the outer third). No such image: the negative is dropped.
  twins     a present question of a rule-answered kind (twin_kinds) -> the same string, ABSENT, on an image without any
            category the string names or the target belongs to. count_pos ("the two dogs") prefers an image where the
            category is present with a different count, which is the harder negative.
  zh        a present Chinese RefCOCO translation (zh_kinds) -> the same string, ABSENT, on an image without the target's
            category or any category whose name occurs in the string.
  near      neg_near "a toaster" / "一台烤面包机" (near_kinds) -> the same string, PRESENT, on an image whose only toaster is
            reliable. No such image: the negative is dropped, as for count and position.

A word that can name several categories ("bag", "glass", "bike", "瓶") rules ALL of them out of an absent twin's image.
Category-free kinds (generic_kinds: "something to sit on") also rule out the target's COCO supercategory and its
confusable classes (assemble.NEAR), since the description may fit a couch or a bench as well as the chair it was
written for.

Images come from the COCO instances file minus every held-out evaluation image, and never equal the source image.
Output: the input records with dropped negatives removed, then the new records (image_id, file, W, H, questions).

python -m glance_data.twins.build --mix MIX.jsonl --coco instances_train2014.json --eval-ids eval_image_ids.json \
    --out out/twins [--config configs/twins_v14.json]
"""
import argparse
import collections
import json
import os
import random
import re

from glance_data.assemble import NEAR
from glance_data.twins.config import load_config
from glance_data.twins.lexicon import EN, EN_PL, IS_ZH, NAMES, NUM, ZH, ZH_NUM

KIND_COUNT_POS, KIND_POS_POS, KIND_ZH_NEG, KIND_TWIN_NEG = "mp_count_pos", "mp_pos_pos", "mp_zh_neg", "mp_twin_neg"
KIND_NEAR_POS = "mp_near_pos"
SRC = "aug_twin"


# ---------------------------------------------------------------------------------------------- text rules
def name_maps():
    """-> (English surface form -> {categories}, Chinese name -> {categories}). A word keeps every category it can name."""
    en, zh = collections.defaultdict(set), collections.defaultdict(set)
    for c in EN:
        for w in [EN[c], EN_PL[c]] + NAMES.get(c, []):
            if IS_ZH.search(w):
                zh[w].add(c)
            else:
                for v in (w, w + "s", w + "es", "piece of " + w, "pieces of " + w, w[:-1] + "ies" if w.endswith("y") else w):
                    en[v].add(c)
        zh[ZH[c]].add(c)
    return dict(en), dict(zh)


EN_ALL, ZH_ALL = name_maps()
EN_WORDS = sorted(EN_ALL, key=len, reverse=True)
ZH_WORDS = sorted(ZH_ALL, key=len, reverse=True)


def first_claim(names):
    """For parsing a template: a word names the category of its question-facing name (EN/EN_PL/ZH) if it is one,
    else the single category it names; words that stay ambiguous do not parse."""
    own = {}
    for c in EN:
        for w in (EN[c], EN_PL[c], ZH[c]):
            own.setdefault(w, c)
    return {w: own.get(w) or next(iter(cs)) for w, cs in names.items() if w in own or len(cs) == 1}


EN_MAP, ZH_MAP = first_claim(EN_ALL), first_claim(ZH_ALL)
EN_NUM = {v: k for k, v in NUM.items()}
ZH_NUMR = {v: k for k, v in ZH_NUM.items()} | {"二": 2}
RE_CNT_EN = re.compile(r"^(?:the |all )?(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve) (.+)$")
RE_CNT_ZH = re.compile(r"^(十一|十二|两|二|三|四|五|六|七|八|九|十)(.)(.+)$")   # number, measure word, name
RE_POS_EN = re.compile(r"^the (.+) on the (left|right)(?: side)?$")
RE_POS_ZH = re.compile(r"^(左|右)边的(.+)$")
RE_NEAR_EN = re.compile(r"^an? (.+)$")
RE_NEAR_ZH = re.compile(r"^一(.)(.+)$")                 # one, measure word, name


def cats_in(text):
    """COCO categories named in `text` (zh: substring; en: whole word, plural and "piece of" forms)."""
    t = " " + text.lower() + " "
    out = set().union(*[ZH_ALL[w] for w in ZH_WORDS if w in text])
    out |= set().union(*[EN_ALL[w] for w in EN_WORDS if f" {w} " in t or f" {w}," in t or f" {w}." in t])
    return out


def parse_count(text):
    """"two clocks" / "the three dogs" / "两个钟" -> (n, category), else None."""
    m = RE_CNT_EN.match(text)
    if m and m.group(2) in EN_MAP:
        return EN_NUM[m.group(1)], EN_MAP[m.group(2)]
    m = RE_CNT_ZH.match(text)
    if m:
        for nm in (m.group(3), m.group(2) + m.group(3)):   # the measure word can be part of the name (个人 -> 人 fails)
            if nm in ZH_MAP:
                return ZH_NUMR[m.group(1)], ZH_MAP[nm]
    return None


def parse_pos(text):
    """"the cup on the left" / "左边的杯子" -> ("left", "cup"), else None."""
    m = RE_POS_EN.match(text)
    if m and m.group(1) in EN_MAP:
        return m.group(2), EN_MAP[m.group(1)]
    m = RE_POS_ZH.match(text)
    if m and m.group(2) in ZH_MAP:
        return {"左": "left", "右": "right"}[m.group(1)], ZH_MAP[m.group(2)]
    return None


def parse_near(text):
    """"a toaster" / "an apple" / "一台烤面包机" -> category, else None."""
    m = RE_NEAR_EN.match(text)
    if m and m.group(1) in EN_MAP:
        return EN_MAP[m.group(1)]
    m = RE_NEAR_ZH.match(text)
    if m:
        for nm in (m.group(2), m.group(1) + m.group(2)):
            if nm in ZH_MAP:
                return ZH_MAP[nm]
    return None


def kind_of(q):
    """Glance-COCO records use `kind`; prepared training mixtures use `qtype`."""
    return q.get("kind") or q.get("qtype") or ""


# ---------------------------------------------------------------------------------------------- image rules
def reliable(o, W, H, cfg):
    """Usable as a positive: not crowd, big enough, and not a small sliver cut off at the border."""
    x0, y0, x1, y1 = o["box"]
    area = (x1 - x0) * (y1 - y0) / (W * H)
    e = cfg.edge_px
    edge = x0 <= e or y0 <= e or x1 >= W - e or y1 >= H - e
    return not o["crowd"] and area >= cfg.min_area and not (edge and area < cfg.edge_min_area)


def rbox(o):
    return [round(v, 1) for v in o["box"]]


def load_coco(path):
    """COCO instances json -> (images {id: dict(W, H, file)}, by {id: {category name: [dict(box xyxy, crowd)]}},
    supercat {category name: supercategory name})."""
    coco = json.load(open(path))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    supercat = {c["name"]: c.get("supercategory", c["name"]) for c in coco["categories"]}
    images = {im["id"]: dict(W=im["width"], H=im["height"], file=im["file_name"]) for im in coco["images"]}
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for an in coco["annotations"]:
        x, y, w, h = an["bbox"]
        by[an["image_id"]][cats[an["category_id"]]].append(dict(box=[x, y, x + w, y + h], crowd=bool(an["iscrowd"])))
    return images, by, supercat


def load_eval_ids(path):
    """A json list of image ids, or a dict of such lists (one per benchmark)."""
    d = json.load(open(path))
    return set(d) if isinstance(d, list) else set().union(*[set(v) for v in d.values()])


class Pools:
    """Candidate images per pairing rule, drawn round-robin from a seeded shuffle so that pairs spread over images."""

    def __init__(self, images, by, held, cfg, rng, supercat=None):
        self.cfg, self.by, self.rng = cfg, by, rng
        self.supercat = supercat or {}
        self.pool = [i for i in images if i not in held and i in by]
        self.count = collections.defaultdict(list)   # (cat, n)    -> images with exactly n reliable instances
        self.side = collections.defaultdict(list)    # (cat, side) -> images whose only instance is on that side
        self.sole = collections.defaultdict(list)    # cat         -> images whose only instance is reliable
        self.clean = collections.defaultdict(list)   # cat         -> [(image, n)]: every instance reliable, n countable
        piles = set(cfg.piles)
        for iid in self.pool:
            W, H = images[iid]["W"], images[iid]["H"]
            for c, lst in by[iid].items():
                n = len(lst)
                if c not in piles and all(reliable(o, W, H, cfg) for o in lst):
                    self.clean[c].append((iid, n))
                    if cfg.count_min <= n <= cfg.count_max:
                        self.count[(c, n)].append(iid)
                if n == 1 and reliable(lst[0], W, H, cfg):
                    self.sole[c].append(iid)
                    cx = (lst[0]["box"][0] + lst[0]["box"][2]) / 2
                    if cx < W * cfg.side_frac:
                        self.side[(c, "left")].append(iid)
                    elif cx > W * (1 - cfg.side_frac):
                        self.side[(c, "right")].append(iid)
        for ix in (self.count, self.side, self.sole):
            for v in ix.values():
                rng.shuffle(v)
        self.ptr = collections.Counter()

    def take(self, ix, key, avoid):
        """Next image for `key`, skipping the source image; None when there is none."""
        L = ix.get(key) or []
        for _ in range(len(L)):
            iid = L[self.ptr[key] % len(L)]
            self.ptr[key] += 1
            if iid != avoid:
                return iid
        return None

    def wrong_count(self, cat, n, avoid):
        """A random image where `cat` is present with a count other than n, and every instance is reliable (a crowd
        region or a cut-off speck would make the count, and so the label, uncertain); None when there is none."""
        L = self.clean.get(cat, [])
        for _ in range(self.cfg.absent_tries if L else 0):
            iid, k = L[self.rng.randrange(len(L))]
            if k != n and iid != avoid:
                return iid
        return None

    def widen(self, cats):
        """Categories a category-free description may also fit: same COCO supercategory, or confusable (NEAR)."""
        out = set(cats)
        for c in cats:
            out |= set(NEAR.get(c, []))
            if c in self.supercat:
                out |= {d for d, s in self.supercat.items() if s == self.supercat[c]}
        return out

    def without(self, bad, avoid):
        """A random image containing none of the categories in `bad` (annotated, crowd included)."""
        for _ in range(self.cfg.absent_tries):
            iid = self.pool[self.rng.randrange(len(self.pool))]
            if iid != avoid and not (bad & set(self.by[iid])):
                return iid
        return None


def question(text, lang, kind, boxes, twin_of=None):
    q = dict(text=text, lang=lang, kind=kind, src=SRC, present=bool(boxes), box=boxes[0] if len(boxes) == 1 else None,
             boxes=boxes or None, n_targets=len(boxes))
    if twin_of:
        q["twin_of"] = twin_of
    return q


def target_cats(q, src_cats, cfg):
    """Categories the absent twin's image must not contain: those named in the text, the referred category, and the
    categories of annotations that coincide with the target boxes on the source image."""
    bad = set(cats_in(q["text"]))
    if q.get("ref_cat"):
        bad.add(q["ref_cat"])
    targets = [q["box"]] if q.get("box") else []
    targets += q.get("boxes") or []
    for c, lst in src_cats.items():
        if any(max(abs(u - v) for u, v in zip(rbox(o), b)) < cfg.box_match_px for o in lst for b in targets):
            bad.add(c)
    return bad


def build(records, pools, cfg):
    """records: iterable of input records. -> (kept records, {image_id: [new questions]}, stats Counter)."""
    by = pools.by
    new = collections.defaultdict(list)
    st = collections.Counter()
    kept, absent_src = [], []
    twin_kinds, zh_kinds, near_kinds, generic = set(cfg.twin_kinds), set(cfg.zh_kinds), set(cfg.near_kinds), set(cfg.generic_kinds)
    for r in records:
        iid0 = r.get("image_id")
        keep = []
        for q in r.get("questions", []):
            k = kind_of(q)
            if k == "neg_count":
                p = parse_count(q["text"])
                iid = pools.take(pools.count, (p[1], p[0]), iid0) if p else None
                if iid is None:
                    st["count_neg_dropped" if p else "count_neg_unparsed"] += 1
                    continue
                n, c = p
                new[iid].append(question(q["text"], q.get("lang"), KIND_COUNT_POS, [rbox(o) for o in by[iid][c]]))
                st["count_pairs"] += 1
            elif k == "neg_position":
                p = parse_pos(q["text"])
                iid = pools.take(pools.side, (p[1], p[0]), iid0) if p else None
                if iid is None:
                    st["pos_neg_dropped" if p else "pos_neg_unparsed"] += 1
                    continue
                new[iid].append(question(q["text"], q.get("lang"), KIND_POS_POS, [rbox(by[iid][p[1]][0])]))
                st["pos_pairs"] += 1
            elif k in near_kinds:
                c = parse_near(q["text"])
                iid = pools.take(pools.sole, c, iid0) if c else None
                if iid is None:
                    st["near_neg_dropped" if c else "near_neg_unparsed"] += 1
                    continue
                new[iid].append(question(q["text"], q.get("lang"), KIND_NEAR_POS, [rbox(by[iid][c][0])]))
                st["near_pairs"] += 1
            elif q.get("present") and k in zh_kinds:
                absent_src.append((iid0, q, KIND_ZH_NEG))
            elif q.get("present") and k in twin_kinds:
                absent_src.append((iid0, q, KIND_TWIN_NEG))
            keep.append(q)
        if keep:
            kept.append({**r, "questions": keep})
    # absent twins are drawn after all positives exist, in a second pass with their own random draws
    for iid0, q, tag in absent_src:
        k = kind_of(q)
        if tag == KIND_TWIN_NEG and k == "count_pos" and q.get("n_targets"):
            cs = cats_in(q["text"])
            iid = pools.wrong_count(next(iter(cs)), q["n_targets"], iid0) if len(cs) == 1 else None
            if iid is not None:
                new[iid].append(question(q["text"], q.get("lang"), tag, [], twin_of=k))
                st["twin_count_wrongn"] += 1
                continue
        bad = target_cats(q, by.get(iid0, {}), cfg)
        if k in generic:
            bad = pools.widen(bad)
        if not bad:
            st[f"{tag}_no_category"] += 1
            continue
        iid = pools.without(bad, iid0)
        if iid is None:
            st[f"{tag}_no_image"] += 1
            continue
        new[iid].append(question(q["text"], q.get("lang") or "zh", tag, [], twin_of=k))
        st[tag] += 1
    return kept, new, st


def new_records(new, images, cfg):
    step = cfg.max_questions_per_record or None
    for iid, qs in new.items():
        im = images[iid]
        for i in range(0, len(qs), step or len(qs)):
            yield dict(image_id=iid, file=im["file"], W=im["W"], H=im["H"], src=SRC, questions=qs[i:i + step if step else None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True, help="input records (Glance-COCO aug_*.jsonl or a prepared training mix)")
    ap.add_argument("--coco", required=True, help="COCO instances json for the candidate images, e.g. instances_train2014.json")
    ap.add_argument("--eval-ids", required=True, help="json list (or dict of lists) of held-out image ids, never used")
    ap.add_argument("--out", required=True, help="output directory: train.jsonl + stats_twins.json")
    ap.add_argument("--config", help="json overrides for TwinConfig (see configs/twins_v14.json)")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config, seed=a.seed)
    images, by, supercat = load_coco(a.coco)
    pools = Pools(images, by, load_eval_ids(a.eval_ids), cfg, random.Random(cfg.seed), supercat)
    kept, new, st = build((json.loads(line) for line in open(a.mix)), pools, cfg)
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/train.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in new_records(new, images, cfg):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            st["new_records"] += 1
    st["records_total"] = len(kept) + st["new_records"]
    st["candidate_images"] = len(pools.pool)     # depends on --coco: record it so a build can be compared
    json.dump(dict(st), open(f"{a.out}/stats_twins.json", "w"), indent=1)
    print(json.dumps(dict(st)))


if __name__ == "__main__":
    main()
