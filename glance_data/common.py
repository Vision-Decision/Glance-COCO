"""Shared configuration, loaders and text heuristics.

Paths come from environment variables (see .env.example):
  JEV_DATA      raw sources (default ~/experiments/jevbox_20261117/data)
  JEV_PREP      the prepared records: train.jsonl, eval_*.jsonl (default ~/experiments/jevbox_20261117/v2/data/prep)
  GLANCE_OUT    where this pipeline writes (default ./out)

Sources under JEV_DATA:
  DATA/{refcoco,refcocoplus,refcocog}/data/<split>-*.parquet   HF jxu124/* (one row per referred object, with ann_id)
  DATA/grefcoco/grefs(unc).json                                 gRefCOCO (multi-target / no-target expressions)
  DATA/grefcoco/instances.json                                  COCO 2014 instances of the 19,994 RefCOCO images
"""
import collections
import glob
import json
import os
import re

DATA = os.environ.get("JEV_DATA", os.path.expanduser("~/experiments/jevbox_20261117/data"))
PREP = os.environ.get("JEV_PREP", os.path.expanduser("~/experiments/jevbox_20261117/v2/data/prep"))
OUT = os.environ.get("GLANCE_OUT", "out")
IMG_ROOT = f"{DATA}/coco_train2014"   # COCO_train2014_<id>.jpg (RefCOCO family images)
SPLITS = {"refcoco": ["train", "validation", "test", "testB"], "refcocoplus": ["train", "validation", "test", "testB"],
          "refcocog": ["train", "validation", "test"]}
SPLIT_NAME = {"train": "train", "validation": "val", "test": "testA", "testB": "testB"}  # refcocog "test" -> "test"


def norm_text(s):
    return re.sub(r"\s+", " ", s.strip()).rstrip(".").strip()


def load_refs():
    """-> list of dict(ds, split, image_id, ann_id, cat, box xyxy, sents[list of str])."""
    import pandas as pd
    out = []
    for ds, sps in SPLITS.items():
        for sp in sps:
            path = glob.glob(f"{DATA}/{ds}/data/{sp}-*.parquet")[0]
            df = pd.read_parquet(path, columns=["image_id", "ann_id", "category_id", "bbox", "sentences", "split"])
            name = "test" if (ds == "refcocog" and sp == "test") else SPLIT_NAME[sp]
            for r in df.itertuples():
                out.append(dict(ds=ds, split=name, image_id=int(r.image_id), ann_id=int(r.ann_id), cat=int(r.category_id),
                                box=[float(x) for x in r.bbox], sents=[norm_text(s["sent"]) for s in r.sentences]))
    return out


def load_grefs():
    return json.load(open(f"{DATA}/grefcoco/grefs(unc).json"))


def load_instances(cache=None):
    """-> (images {id: dict(width, height)}, anns_by_image {id: [ann]}, cats {coco id: name}); ann bbox is xyxy.
    gRefCOCO's instances.json covers the 19,994 RefCOCO(unc) images (with iscrowd); the rest (RefCOCOg-only images,
    COCO 2017 extras) come from the COCO 2017 parquet (same annotation ids; contiguous category index -> COCO id)."""
    cache = cache or f"{OUT}/instances_all.json"
    if os.path.exists(cache):
        d = json.load(open(cache))
        return ({int(k): v for k, v in d["images"].items()}, collections.defaultdict(list, {int(k): v for k, v in d["anns"].items()}),
                {int(k): v for k, v in d["cats"].items()})
    d = json.load(open(f"{DATA}/grefcoco/instances.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    contig = sorted(cats)                                   # 0..79 -> COCO category id
    images = {im["id"]: dict(width=im["width"], height=im["height"]) for im in d["images"]}
    by_img = collections.defaultdict(list)
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        by_img[a["image_id"]].append(dict(id=a["id"], cat=a["category_id"], box=[x, y, x + w, y + h], area=a["area"],
                                          crowd=a["iscrowd"]))
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(f"{DATA}/coco2017/data/*.parquet")):
        t = pq.read_table(f, columns=["image_id", "width", "height", "objects"]).to_pylist()
        for r in t:
            iid = int(r["image_id"])
            if iid in images:
                continue
            images[iid] = dict(width=r["width"], height=r["height"])
            o = r["objects"]
            for k in range(len(o["bbox_id"])):
                by_img[iid].append(dict(id=int(o["bbox_id"][k]), cat=contig[int(o["category"][k])],
                                        box=[float(v) for v in o["bbox"][k]], area=float(o["area"][k]), crowd=0))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(dict(images=images, anns=by_img, cats=cats), open(cache, "w"))
    return images, by_img, cats


# ---------------------------------------------------------------------------------------------- text heuristics
PERSON = """person people man men woman women guy guys girl girls boy boys kid kids child children lady ladies baby
 player players skier surfer snowboarder batter catcher pitcher umpire dude gentleman adult teen teenager toddler
 chef cop officer rider biker skateboarder tennis""".split()
SYN = {  # COCO category -> extra head nouns
    "person": PERSON, "bicycle": ["bike", "bikes"], "motorcycle": ["motorbike", "motorbikes", "bike", "scooter"],
    "airplane": ["plane", "planes", "jet", "aircraft"], "tv": ["television", "screen", "monitor", "tvs"],
    "couch": ["sofa", "sofas", "couches", "loveseat"], "dining table": ["table", "tables", "desk"],
    "cell phone": ["phone", "phones", "cellphone"], "cup": ["mug", "mugs", "glass", "glasses", "cups"],
    "donut": ["doughnut", "donuts", "doughnuts"], "hot dog": ["hotdog", "hotdogs"], "potted plant": ["plant", "plants"],
    "sports ball": ["ball", "balls"], "teddy bear": ["teddy", "bear"], "laptop": ["computer", "laptops"],
    "car": ["cars", "vehicle", "suv", "van", "taxi"], "truck": ["trucks", "pickup"], "bus": ["buses"],
    "cow": ["cows", "cattle", "bull", "calf"], "sheep": ["lamb", "lambs"], "elephant": ["elephants"],
    "zebra": ["zebras"], "giraffe": ["giraffes"], "horse": ["horses", "pony"], "dog": ["dogs", "puppy"],
    "cat": ["cats", "kitten", "kitty"], "bird": ["birds", "duck", "seagull", "pigeon"], "chair": ["chairs", "seat"],
    "bed": ["beds"], "bench": ["benches"], "umbrella": ["umbrellas"], "pizza": ["slice", "pizzas"],
    "sandwich": ["sandwiches", "burger"], "bowl": ["bowls"], "bottle": ["bottles"], "boat": ["boats", "ship"],
    "banana": ["bananas"], "broccoli": ["broccolis"], "cake": ["cakes", "dessert"], "suitcase": ["luggage", "bag"],
    "handbag": ["purse", "bag"], "backpack": ["bag"], "skis": ["ski"], "surfboard": ["board", "surfboards"],
    "skateboard": ["board", "skateboards"], "train": ["trains", "locomotive"], "vase": ["vases"], "book": ["books"],
    "clock": ["clocks"], "orange": ["oranges"], "apple": ["apples"], "carrot": ["carrots"], "knife": ["knives"],
    "spoon": ["spoons"], "fork": ["forks"], "wine glass": ["glass", "wineglass"], "toilet": ["toilets"],
    "sink": ["sinks"], "refrigerator": ["fridge"], "oven": ["stove"], "keyboard": ["keyboards"], "mouse": ["mice"],
}
SPATIAL = set("""left right middle center centre top bottom front back behind closest closer nearest near far farthest
 furthest first second third fourth last upper lower corner side edge background foreground above below under
 beside next between leftmost rightmost topmost farther further rear lefter righter""".split())
APPEAR = set("""red blue green yellow black white gray grey brown pink purple orange tan beige dark light striped plaid
 shirt shirts jacket coat hat cap dress pants jeans shorts sweater hoodie uniform glasses helmet skirt suit tie vest
 scarf shoes boots jersey hair bald beard blonde blond big bigger biggest small smaller smallest tall taller tallest
 short large larger largest little young old older wooden metal plastic glass round square long""".split())
RELATION = set("""holding wearing with sitting standing walking riding eating looking playing carrying hitting
 throwing swinging reaching leaning lying laying sleeping talking drinking cutting pointing facing watching using""".split())
FUNCTION_PAT = re.compile(r"\b(something|thing|things|stuff|object)\b|\bused (to|for)\b|\bto (eat|drink|sit|ride|wear)\b")


def head_nouns(cat_names):
    """category id -> set of words that name it (lower-case, single tokens; multi-word names add their last word)."""
    out = {}
    for cid, name in cat_names.items():
        w = set(SYN.get(name, [])) | {name, name.split()[-1], name + "s"}
        out[cid] = w
    return out


def facets(text, nouns_all):
    """-> dict of booleans + n words. nouns_all: set of all category head words."""
    toks = re.findall(r"[a-z]+", text.lower())
    t = set(toks)
    return dict(n=len(toks), noun=bool(t & nouns_all), spatial=bool(t & SPATIAL), appear=bool(t & APPEAR),
                relation=bool(t & RELATION), function=bool(FUNCTION_PAT.search(text.lower())))


def mentioned_cats(text, nouns_by_cat):
    t = set(re.findall(r"[a-z]+", text.lower()))
    return {c for c, ws in nouns_by_cat.items() if t & ws}
