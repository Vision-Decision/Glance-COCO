"""Stage 3 (server): turn inventory + LLM generations + verification into training records.

Ground truth is decided by code wherever annotations allow it; the LLM only phrases. Sources:
  llm      ladder (short/medium/long/casual, en+zh), hard negatives (attribute/relation), groups
           -> rule filters (no mark-leak words); whole kinds can be switched off with --drop-kinds after the audit
  function use/role descriptions -> targets = ALL instances of the COCO categories the LLM named (code)
  code     category-only (1 instance -> single, n -> all n), count negatives (more than present),
           position negatives (no instance of that category on that side), near-category negatives (absent COCO class)
  gref     gRefCOCO multi-target human sentences (dropped in v0/v2 training) -> multi-target positives

python -m glance_data.assemble --inv INV --gen GEN/gen.jsonl --out OUT/aug.jsonl [--drop-kinds hneg_relation,casual]
Output: one record per image, same schema as JEV_PREP/train.jsonl plus per-question
        boxes (all targets), n_targets, kind, lang, src ("aug_llm" | "aug_function" | "aug_code" | "aug_gref").
"""
import argparse
import collections
import json
import os
import random
import re

from glance_data.twins.lexicon import ZH_MW

# COCO category -> (English singular, English plural, Chinese)
NAMES = {
    "person": ("person", "people", "人"), "bicycle": ("bicycle", "bicycles", "自行车"), "car": ("car", "cars", "汽车"),
    "motorcycle": ("motorcycle", "motorcycles", "摩托车"), "airplane": ("airplane", "airplanes", "飞机"),
    "bus": ("bus", "buses", "公交车"), "train": ("train", "trains", "火车"), "truck": ("truck", "trucks", "卡车"),
    "boat": ("boat", "boats", "船"), "traffic light": ("traffic light", "traffic lights", "红绿灯"),
    "fire hydrant": ("fire hydrant", "fire hydrants", "消防栓"), "stop sign": ("stop sign", "stop signs", "停车标志"),
    "parking meter": ("parking meter", "parking meters", "停车计时器"), "bench": ("bench", "benches", "长椅"),
    "bird": ("bird", "birds", "鸟"), "cat": ("cat", "cats", "猫"), "dog": ("dog", "dogs", "狗"), "horse": ("horse", "horses", "马"),
    "sheep": ("sheep", "sheep", "羊"), "cow": ("cow", "cows", "牛"), "elephant": ("elephant", "elephants", "大象"),
    "bear": ("bear", "bears", "熊"), "zebra": ("zebra", "zebras", "斑马"), "giraffe": ("giraffe", "giraffes", "长颈鹿"),
    "backpack": ("backpack", "backpacks", "背包"), "umbrella": ("umbrella", "umbrellas", "雨伞"),
    "handbag": ("handbag", "handbags", "手提包"), "tie": ("tie", "ties", "领带"), "suitcase": ("suitcase", "suitcases", "行李箱"),
    "frisbee": ("frisbee", "frisbees", "飞盘"), "skis": ("pair of skis", "pairs of skis", "滑雪板"),
    "snowboard": ("snowboard", "snowboards", "单板滑雪板"), "sports ball": ("ball", "balls", "球"), "kite": ("kite", "kites", "风筝"),
    "baseball bat": ("baseball bat", "baseball bats", "棒球棒"), "baseball glove": ("baseball glove", "baseball gloves", "棒球手套"),
    "skateboard": ("skateboard", "skateboards", "滑板"), "surfboard": ("surfboard", "surfboards", "冲浪板"),
    "tennis racket": ("tennis racket", "tennis rackets", "网球拍"), "bottle": ("bottle", "bottles", "瓶子"),
    "wine glass": ("wine glass", "wine glasses", "酒杯"), "cup": ("cup", "cups", "杯子"), "fork": ("fork", "forks", "叉子"),
    "knife": ("knife", "knives", "刀"), "spoon": ("spoon", "spoons", "勺子"), "bowl": ("bowl", "bowls", "碗"),
    "banana": ("banana", "bananas", "香蕉"), "apple": ("apple", "apples", "苹果"), "sandwich": ("sandwich", "sandwiches", "三明治"),
    "orange": ("orange", "oranges", "橙子"), "broccoli": ("piece of broccoli", "pieces of broccoli", "西兰花"),
    "carrot": ("carrot", "carrots", "胡萝卜"), "hot dog": ("hot dog", "hot dogs", "热狗"), "pizza": ("pizza", "pizzas", "披萨"),
    "donut": ("donut", "donuts", "甜甜圈"), "cake": ("cake", "cakes", "蛋糕"), "chair": ("chair", "chairs", "椅子"),
    "couch": ("couch", "couches", "沙发"), "potted plant": ("potted plant", "potted plants", "盆栽"), "bed": ("bed", "beds", "床"),
    "dining table": ("table", "tables", "桌子"), "toilet": ("toilet", "toilets", "马桶"), "tv": ("TV", "TVs", "电视"),
    "laptop": ("laptop", "laptops", "笔记本电脑"), "mouse": ("computer mouse", "computer mice", "鼠标"),
    "remote": ("remote", "remotes", "遥控器"), "keyboard": ("keyboard", "keyboards", "键盘"),
    "cell phone": ("cell phone", "cell phones", "手机"), "microwave": ("microwave", "microwaves", "微波炉"),
    "oven": ("oven", "ovens", "烤箱"), "toaster": ("toaster", "toasters", "烤面包机"), "sink": ("sink", "sinks", "水槽"),
    "refrigerator": ("refrigerator", "refrigerators", "冰箱"), "book": ("book", "books", "书"), "clock": ("clock", "clocks", "钟"),
    "vase": ("vase", "vases", "花瓶"), "scissors": ("pair of scissors", "pairs of scissors", "剪刀"),
    "teddy bear": ("teddy bear", "teddy bears", "泰迪熊"), "hair drier": ("hair dryer", "hair dryers", "吹风机"),
    "toothbrush": ("toothbrush", "toothbrushes", "牙刷"),
}
# confusable COCO classes for near-category negatives
NEAR = {
    "dog": ["cat", "horse", "sheep", "bear"], "cat": ["dog", "teddy bear"], "horse": ["cow", "zebra", "dog"],
    "cow": ["horse", "sheep", "elephant"], "sheep": ["cow", "dog"], "zebra": ["horse", "giraffe"], "giraffe": ["zebra", "elephant"],
    "elephant": ["cow", "bear", "giraffe"], "bear": ["dog", "teddy bear"], "teddy bear": ["bear", "cat", "dog"],
    "car": ["truck", "bus", "motorcycle"], "truck": ["car", "bus"], "bus": ["truck", "train", "car"], "train": ["bus", "truck"],
    "motorcycle": ["bicycle", "car"], "bicycle": ["motorcycle"], "boat": ["airplane", "car"], "airplane": ["boat", "bird", "kite"],
    "bird": ["airplane", "kite", "cat"], "kite": ["bird", "umbrella", "frisbee"], "cup": ["wine glass", "bottle", "bowl"],
    "wine glass": ["cup", "bottle"], "bottle": ["cup", "vase", "wine glass"], "bowl": ["cup", "vase"], "vase": ["bottle", "cup"],
    "fork": ["spoon", "knife"], "spoon": ["fork", "knife"], "knife": ["fork", "scissors", "spoon"], "scissors": ["knife"],
    "pizza": ["cake", "sandwich", "donut"], "cake": ["pizza", "donut"], "donut": ["cake", "pizza"], "sandwich": ["hot dog", "pizza"],
    "hot dog": ["sandwich", "pizza"], "banana": ["orange", "apple", "carrot"], "orange": ["apple", "banana"],
    "apple": ["orange", "banana"], "broccoli": ["carrot", "potted plant"], "carrot": ["broccoli", "banana"],
    "chair": ["couch", "bench", "bed"], "couch": ["bed", "chair", "bench"], "bed": ["couch"], "bench": ["chair", "couch"],
    "tv": ["laptop", "microwave"], "laptop": ["tv", "keyboard", "cell phone"], "keyboard": ["laptop", "remote"],
    "cell phone": ["remote", "laptop"], "remote": ["cell phone", "mouse"], "mouse": ["remote", "cell phone"],
    "oven": ["microwave", "refrigerator"], "microwave": ["oven", "tv", "toaster"], "refrigerator": ["oven", "microwave"],
    "sink": ["toilet", "bowl"], "toilet": ["sink"], "umbrella": ["kite"], "handbag": ["backpack", "suitcase"],
    "backpack": ["handbag", "suitcase"], "suitcase": ["backpack", "handbag"], "skis": ["snowboard", "skateboard"],
    "snowboard": ["skis", "surfboard", "skateboard"], "surfboard": ["skateboard", "snowboard", "boat"],
    "skateboard": ["surfboard", "snowboard"], "tennis racket": ["baseball bat"], "baseball bat": ["tennis racket"],
    "sports ball": ["frisbee"], "frisbee": ["sports ball", "kite"], "clock": ["tv"], "book": ["laptop"],
    "potted plant": ["vase", "broccoli"], "person": ["teddy bear", "dog"],
}
NUM_EN = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
NUM_ZH = {2: "两", 3: "三", 4: "四", 5: "五", 6: "六"}


SPATIAL_EN = set("""left right middle center centre top bottom front back behind edge corner side far near nearest
 closest leftmost rightmost upper lower foreground background above below under beside next first second third last""".split())
SPATIAL_ZH = ["左", "右", "中间", "中央", "正中", "上方", "下方", "上面", "下面", "前", "后", "边缘", "角", "旁", "第"]


HNEG_KINDS = {"hneg_attribute", "hneg_relation"}


def spatial_sig(text, lang):
    if lang == "zh":
        return tuple(w for w in SPATIAL_ZH if w in text)
    return tuple(sorted(set(re.findall(r"[a-z]+", text.lower())) & SPATIAL_EN))


def q(text, lang, kind, src, boxes):
    present = bool(boxes)
    return dict(text=text, lang=lang, kind=kind, src=src, present=present, box=boxes[0] if len(boxes) == 1 else None,
                boxes=boxes if present else None, n_targets=len(boxes))


def code_items(rec, rng, min_area=0.005):
    """min_area: a category whose largest instance covers less than this share of the image gets no
    category-only query (slivers at the border, tiny background objects)."""
    W, H = rec["W"], rec["H"]
    by_cat = collections.defaultdict(list)
    for o in rec["instances"]:
        by_cat[o["cat"]].append(o)
    crowd = set(rec["crowd_cats"])
    out = []
    for cat, objs in by_cat.items():
        if cat in crowd or cat not in NAMES:
            continue
        en, pl, zh = NAMES[cat]
        boxes = [o["box"] for o in objs]
        n = len(objs)
        big = max((o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) for o in objs) >= min_area * W * H
        # category-only: one instance -> that one; several -> all of them
        if not big:
            pass
        elif n == 1:
            out += [q(f"the {en}", "en", "cat_only", "aug_code", boxes), q(zh, "zh", "cat_only", "aug_code", boxes)]
        elif n <= 8:
            out += [q(rng.choice([f"all the {pl}", f"the {pl}", f"every {en}"]), "en", "cat_all", "aug_code", boxes),
                    q(rng.choice([f"所有的{zh}", f"全部{zh}"]), "zh", "cat_all", "aug_code", boxes)]
        # count negative: more than are present
        if n <= 5:
            k = n + rng.choice([1, 2]) if n >= 2 else 2
            k = min(k, 6)
            out += [q(f"{NUM_EN[k]} {pl}", "en", "neg_count", "aug_code", []),
                    q(f"{NUM_ZH[k]}{ZH_MW.get(cat, '个')}{zh}", "zh", "neg_count", "aug_code", [])]
        # position negative: no instance of this category on that side (centre-based, with margin)
        cx = [((o["box"][0] + o["box"][2]) / 2) / W for o in objs]
        if min(cx) > 0.6:
            out += [q(f"the {en} on the left", "en", "neg_position", "aug_code", []), q(f"左边的{zh}", "zh", "neg_position", "aug_code", [])]
        elif max(cx) < 0.4:
            out += [q(f"the {en} on the right", "en", "neg_position", "aug_code", []), q(f"右边的{zh}", "zh", "neg_position", "aug_code", [])]
    # near-category negative: a confusable COCO class with zero instances
    present = set(by_cat)
    for cat in list(present):
        cands = [c for c in NEAR.get(cat, []) if c not in present and c not in crowd]
        if cands and rng.random() < 0.6:
            c = rng.choice(cands)
            en, pl, zh = NAMES[c]
            out += [q(f"a {en}" if not en.startswith(("a", "e", "i", "o", "u")) else f"an {en}", "en", "neg_near", "aug_code", []),
                    q(f"一{ZH_MW.get(c, '个')}{zh}", "zh", "neg_near", "aug_code", [])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-kinds", default="", help="comma-separated kinds to leave out (e.g. those that fail the audit)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    inv = {json.loads(line)["image_id"]: json.loads(line) for line in open(f"{a.inv}/inventory.jsonl")}
    gens = {json.loads(line)["image_id"]: json.loads(line) for line in open(a.gen)}
    drop = set(filter(None, a.drop_kinds.split(",")))
    stats = collections.Counter()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for iid, rec in inv.items():
            rng = random.Random(a.seed + iid)
            marks = {m["mark"]: m for m in rec["marked"]}
            by_cat = collections.defaultdict(list)
            for o in rec["instances"]:
                by_cat[o["cat"]].append(o)
            out = []
            g = gens.get(iid)
            medium = {}                                  # (mark, lang) -> medium description, for the hard-negative check
            for o in (g["gen"].get("objects", []) if g else []):
                medium[(o.get("mark"), "en")] = o.get("medium") or ""
                medium[(o.get("mark"), "zh")] = o.get("zh_medium") or ""
            for qq in (g["queries"] if g else []):
                kind = qq["kind"]
                if kind.startswith("hneg_") and kind not in HNEG_KINDS:   # the LLM sometimes invents an edit type
                    kind = "hneg_relation"
                if kind in drop:
                    continue
                if qq["intent"] == "code":              # function: all instances of the named categories
                    cats_ = qq.get("categories") or []
                    if any(c in rec["crowd_cats"] for c in cats_):
                        stats["drop:function_crowd"] += 1
                        continue
                    boxes = [o["box"] for c in cats_ for o in by_cat.get(c, [])]
                    if boxes:
                        out.append(q(qq["text"], qq["lang"], "function", "aug_function", boxes))
                    continue
                if qq.get("leak"):
                    stats["drop:mark_leak"] += 1
                    continue
                if kind.startswith("hneg_"):              # an attribute/relation edit must keep the position words
                    src_txt = medium.get((qq.get("mark"), qq["lang"]), "")
                    if spatial_sig(qq["text"], qq["lang"]) != spatial_sig(src_txt, qq["lang"]):
                        stats["drop:hneg_position_changed"] += 1
                        continue
                it = qq["intent"]
                boxes = [] if it == "none" else [marks[m]["box"] for m in it]
                out.append(q(qq["text"], qq["lang"], kind, "aug_llm", boxes))
            if "code" not in drop:
                out += code_items(rec, rng)
            for gm in (rec["gref_multi"] if "gref_multi" not in drop else []):
                ids = set(gm["ann_ids"])
                boxes = [o["box"] for o in rec["instances"] if o["ann_id"] in ids]
                if len(boxes) == len(ids):
                    out.append(q(gm["text"], "en", "gref_multi", "aug_gref", boxes))
            seen, uniq = set(), []
            for x in out:
                k = x["text"].lower()
                if k not in seen:
                    seen.add(k); uniq.append(x)
            for x in uniq:
                stats[f"out:{x['src']}:{x['kind']}"] += 1
            f.write(json.dumps(dict(image_id=iid, file=rec["file"], W=rec["W"], H=rec["H"], questions=uniq),
                               ensure_ascii=False) + "\n")
    json.dump(dict(sorted(stats.items())), open(a.out.replace(".jsonl", "_stats.json"), "w"), indent=1)
    for k, v in sorted(stats.items()):
        print(f"{k:40s} {v}")


if __name__ == "__main__":
    main()
