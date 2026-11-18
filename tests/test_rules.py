"""Rule-based parts of the pipeline: no data, GPU or network needed."""
import json
import random
import subprocess
import sys

from glance_data.assemble import code_items, spatial_sig
from glance_data.common import facets, head_nouns
from glance_data.generate import collect_queries

W, H = 640, 480


def inst(i, cat, box, crowd=0):
    return dict(ann_id=i, cat=cat, box=box, crowd=crowd)


def record(instances, marked=(), crowd_cats=()):
    counts = {}
    for o in instances:
        counts[o["cat"]] = counts.get(o["cat"], 0) + 1
    return dict(image_id=1, file="x.jpg", W=W, H=H, counts=counts, crowd_cats=list(crowd_cats),
                marked=list(marked), instances=instances, gref_multi=[], gref_none=[])


def test_spatial_sig_en_zh():
    assert spatial_sig("man in a black suit on the left", "en") == ("left",)
    assert spatial_sig("man in a white suit on the left", "en") == spatial_sig("man in a black suit on the left", "en")
    assert spatial_sig("zebra on the far right edge", "en") != spatial_sig("zebra on the far left edge", "en")
    assert spatial_sig("左边穿黑色西装的男子", "zh") != spatial_sig("右边穿黑色西装的男子", "zh")


def test_code_items_counts_positions_and_tiny_objects():
    dogs = [inst(1, "dog", [500, 100, 600, 300]), inst(2, "dog", [420, 120, 520, 330])]   # both on the right
    cup = inst(3, "cup", [10, 10, 14, 14])                                                  # sliver: < 0.5% of image
    items = code_items(record(dogs + [cup]), random.Random(0))
    by_kind = {}
    for q in items:
        by_kind.setdefault(q["kind"], []).append(q)
    # "all the dogs" -> both boxes; no category-only query for the tiny cup
    assert all(q["n_targets"] == 2 and q["present"] for q in by_kind["cat_all"])
    assert not any("cup" in q["text"] or q["text"] == "杯子" for q in by_kind.get("cat_only", []))
    # count negative asks for more than are present
    assert all(not q["present"] for q in by_kind["neg_count"])
    assert any(t in by_kind["neg_count"][0]["text"] for t in ("three", "four"))
    # every dog is on the right -> "the dog on the left" is absent
    assert {q["text"] for q in by_kind["neg_position"]} >= {"the dog on the left", "左边的狗"}


def test_code_items_skip_crowd_categories():
    people = [inst(1, "person", [0, 0, 300, 400]), inst(2, "person", [300, 0, 600, 400], crowd=1)]
    items = code_items(record(people, crowd_cats=["person"]), random.Random(0))
    assert not any(q["kind"] in ("cat_only", "cat_all", "neg_count") for q in items)


def test_near_category_negatives_are_absent_classes():
    rec = record([inst(1, "dog", [0, 0, 300, 300])])
    for seed in range(20):
        for q in code_items(rec, random.Random(seed)):
            if q["kind"] == "neg_near":
                assert "dog" not in q["text"] and not q["present"]


def test_collect_queries_filters():
    rec = record([inst(1, "dog", [0, 0, 10, 10])], marked=[dict(mark=1, cat="dog", box=[0, 0, 10, 10], human=[])])
    gen = {"objects": [
        {"mark": 1, "short": "brown dog", "medium": "brown dog in the box", "zh_short": "棕色的狗",
         "hard_negatives": [{"text": "black dog", "zh": "黑色的狗", "edit": "attribute"}]},
        {"mark": 9, "short": "ghost object"},                       # not a valid mark
    ], "groups": [{"marks": [1], "text": "just one"}],              # a group needs >= 2 marks
        "functions": [{"text": "a pet", "zh": "宠物", "categories": ["dog", "unicorn"]}]}
    qs = collect_queries(rec, gen)
    texts = {q["text"]: q for q in qs}
    assert "ghost object" not in texts and "just one" not in texts
    assert texts["brown dog in the box"]["leak"] is True          # mentions the drawn box
    assert texts["black dog"]["intent"] == "none"
    assert texts["a pet"]["categories"] == ["dog"]                # categories not in the image are removed


def test_assemble_end_to_end(tmp_path):
    marks = [dict(mark=1, ann_id=1, cat="person", box=[10, 10, 200, 400], box_norm=[0, 0, 0.3, 0.8], human=["left guy"])]
    rec = record([inst(1, "person", [10, 10, 200, 400]), inst(2, "tie", [80, 100, 110, 200])], marked=marks)
    inv = tmp_path / "inv"
    inv.mkdir()
    (inv / "inventory.jsonl").write_text(json.dumps(rec) + "\n")
    gen = {"objects": [{"mark": 1, "short": "man in a suit", "medium": "man in a black suit on the left",
                        "zh_medium": "左边穿黑色西装的男子",
                        "hard_negatives": [
                            {"text": "man in a white suit on the left", "zh": "左边穿白色西装的男子", "edit": "attribute"},
                            {"text": "man in a black suit on the right", "zh": "右边穿黑色西装的男子", "edit": "attribute"}]}]}
    g = dict(image_id=1, gen=gen, queries=collect_queries(rec, gen))
    (tmp_path / "gen.jsonl").write_text(json.dumps(g, ensure_ascii=False) + "\n")
    out = tmp_path / "aug.jsonl"
    subprocess.run([sys.executable, "-m", "glance_data.assemble", "--inv", str(inv), "--gen",
                    str(tmp_path / "gen.jsonl"), "--out", str(out)], check=True, capture_output=True)
    qs = {q["text"]: q for q in json.loads(out.read_text())["questions"]}
    assert qs["man in a suit"]["present"] and qs["man in a suit"]["box"] == [10, 10, 200, 400]
    assert not qs["man in a white suit on the left"]["present"]          # attribute edit kept
    assert "man in a black suit on the right" not in qs                  # position edit dropped
    assert "右边穿黑色西装的男子" not in qs
    assert qs["the tie"]["n_targets"] == 1                                # rule-made category-only query


def test_facets_heuristics():
    nouns = set().union(*head_nouns({1: "person", 18: "dog"}).values())
    f = facets("the man on the left in a red shirt", nouns)
    assert f["noun"] and f["spatial"] and f["appear"]
    assert not facets("middle left", nouns)["noun"]
