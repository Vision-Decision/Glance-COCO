"""Minimal-pair builder and text-only gate: no data, GPU or network needed."""
import json
import random
import subprocess
import sys

import numpy as np
import pytest

from glance_data.twins.build import Pools, build, cats_in, parse_count, parse_pos, reliable
from glance_data.twins.config import TwinConfig, load_config
from glance_data.twins.gate import auroc, held_out, run

W, H = 600, 400
CFG = TwinConfig()


def ann(cat, box, crowd=False):
    return cat, dict(box=box, crowd=crowd)


def coco(spec):
    """{image_id: [ann(...)]} -> (images, by) as load_coco returns them."""
    images = {i: dict(W=W, H=H, file=f"COCO_train2014_{i:012d}.jpg") for i in spec}
    by = {}
    for i, anns in spec.items():
        by[i] = {}
        for c, o in anns:
            by[i].setdefault(c, []).append(o)
    return images, by


def pools(spec, held=(), seed=0):
    images, by = coco(spec)
    return Pools(images, by, set(held), CFG, random.Random(seed))


def q(text, kind, present, lang="en", boxes=None, **kw):
    boxes = boxes or []
    return dict(text=text, kind=kind, present=present, lang=lang, box=boxes[0] if len(boxes) == 1 else None,
                boxes=boxes or None, n_targets=len(boxes), **kw)


def rec(iid, *qs):
    return dict(image_id=iid, file="x.jpg", W=W, H=H, questions=list(qs))


DOG_L, DOG_R, DOG_M = [20, 100, 140, 300], [460, 100, 580, 300], [250, 100, 350, 300]


def test_parse_templates_en_zh():
    assert parse_count("two dogs") == (2, "dog") and parse_count("the three cats") == (3, "cat")
    assert parse_count("两只狗") == (2, "dog") and parse_count("四个人") == (4, "person")
    assert parse_count("two dogs on a sofa") is None and parse_count("the dog") is None
    assert parse_pos("the cup on the left") == ("left", "cup") and parse_pos("右边的杯子") == ("right", "cup")
    assert parse_pos("the cup on the left of the plate") is None
    assert cats_in("a man holding a cup") == {"person", "cup"} and cats_in("穿红衣服的女人") == {"person"}


def test_reliable_rejects_crowd_tiny_and_border_slivers():
    assert reliable(dict(box=DOG_M, crowd=False), W, H, CFG)
    assert not reliable(dict(box=DOG_M, crowd=True), W, H, CFG)
    assert not reliable(dict(box=[100, 100, 110, 110], crowd=False), W, H, CFG)       # 0.04% of the image
    assert not reliable(dict(box=[0, 100, 40, 150], crowd=False), W, H, CFG)          # 0.8%, touches the border
    assert reliable(dict(box=[0, 100, 120, 250], crowd=False), W, H, CFG)             # 7.5% at the border is fine


def test_count_negative_gets_same_string_positive_or_is_dropped():
    p = pools({1: [ann("dog", DOG_L)],                                   # source: one dog
               2: [ann("dog", DOG_L), ann("dog", DOG_R)],                # exactly two reliable dogs
               3: [ann("book", DOG_L), ann("book", DOG_R)],              # books pile up: never a count positive
               4: [ann("dog", DOG_L), ann("dog", [5, 5, 15, 15])]})      # second dog is a speck: not countable
    kept, new, st = build([rec(1, q("two dogs", "neg_count", False), q("两只狗", "neg_count", False, "zh"),
                               q("two books", "neg_count", False), q("six dogs", "neg_count", False))], p, CFG)
    assert [x["text"] for x in kept[0]["questions"]] == ["two dogs", "两只狗"]      # lone negatives are dropped
    assert st["count_pairs"] == 2 and st["count_neg_dropped"] == 2
    assert set(new) == {2}
    for x in new[2]:
        assert x["present"] and x["n_targets"] == 2 and x["kind"] == "mp_count_pos" and len(x["boxes"]) == 2


def test_position_negative_gets_positive_with_sole_object_on_that_side():
    p = pools({1: [ann("dog", DOG_R)], 2: [ann("dog", DOG_L)], 3: [ann("dog", DOG_M)],
               4: [ann("dog", DOG_L), ann("dog", DOG_R)]})
    kept, new, st = build([rec(1, q("the dog on the left", "neg_position", False),
                               q("右边的狗", "neg_position", False, "zh"))], p, CFG)
    assert st["pos_pairs"] == 1 and st["pos_neg_dropped"] == 1                     # no other image has a sole right dog
    assert list(new) == [2] and new[2][0]["box"] == DOG_L and new[2][0]["text"] == "the dog on the left"
    assert [x["text"] for x in kept[0]["questions"]] == ["the dog on the left"]


def test_missing_category_twins_avoid_every_named_category():
    spec = {1: [ann("dog", DOG_L), ann("person", DOG_R)], 2: [ann("dog", DOG_M)], 3: [ann("person", DOG_M)],
            4: [ann("cat", DOG_M)]}
    src = [q("the dog", "cat_only", True, boxes=[DOG_L]), q("the dog next to the man", "cf_anchor", True, boxes=[DOG_L]),
           q("左边的那只动物", "ref_zh", True, "zh", boxes=[DOG_L]),       # no category word: matched through its box
           q("a black dog", "hneg_attribute", False)]                       # LLM negative: not twinned
    for seed in range(10):
        _, new, st = build([rec(1, *src)], pools(spec, seed=seed), CFG)
        assert st["mp_twin_neg"] == 2 and st["mp_zh_neg"] == 1
        placed = {x["twin_of"]: iid for iid, xs in new.items() for x in xs}
        assert set(placed) == {"cat_only", "cf_anchor", "ref_zh"}
        assert placed["cat_only"] in (3, 4) and placed["ref_zh"] in (3, 4)    # any image without a dog
        assert placed["cf_anchor"] == 4                                        # neither a dog nor a person
        assert all(not x["present"] and x["boxes"] is None for xs in new.values() for x in xs)


def test_count_pos_twin_prefers_wrong_count_image():
    p = pools({1: [ann("dog", DOG_L), ann("dog", DOG_R)], 2: [ann("dog", DOG_M)], 3: [ann("cat", DOG_M)]})
    _, new, st = build([rec(1, q("the two dogs", "count_pos", True, boxes=[DOG_L, DOG_R]))], p, CFG)
    assert st["twin_count_wrongn"] == 1 and list(new) == [2]                # dog present, but one, not two


def test_held_out_images_are_never_used():
    spec = {1: [ann("dog", DOG_R)], 2: [ann("dog", DOG_L)], 3: [ann("cat", DOG_M)],
            4: [ann("dog", DOG_L), ann("dog", DOG_R)]}
    _, new, st = build([rec(1, q("the dog on the left", "neg_position", False), q("two dogs", "neg_count", False),
                            q("the dog", "cat_only", True, boxes=[DOG_R]))], pools(spec, held={2, 3, 4}), CFG)
    assert not new and st["pos_neg_dropped"] == 1 and st["count_neg_dropped"] == 1


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 300), rng.integers(0, 5, 300).astype(float)       # many ties
    pos, neg = s[y == 1], s[y == 0]
    ref = ((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg))
    assert auroc(y, s) == pytest.approx(ref)
    assert auroc([1, 1], [0.2, 0.3]) is None


def memorise(texts, y):
    """A probe that remembers each training string's label: the worst case for an exact-twin leak."""
    seen = dict(zip(texts, y))
    return lambda ts: [seen.get(t, 0.5) for t in ts]


def test_image_split_puts_twins_on_both_sides_and_reads_below_chance():
    # 400 strings, each present on one image and absent (its twin) on another
    recs = [rec(2 * i, q(f"string {i}", "cat_only", True)) for i in range(400)]
    recs += [rec(2 * i + 1, q(f"string {i}", "mp_twin_neg", False, twin_of="cat_only")) for i in range(400)]
    by_text = run(recs, CFG, "text", probe=memorise)["gates"]["S1"]
    by_image = run(recs, CFG, "image", probe=memorise)["gates"]["S1"]
    assert by_text["auroc"] == 0.5                       # both twins held out together: nothing to remember
    assert by_image["auroc"] < 0.2                       # the train twin's label is the test twin's opposite
    mask = held_out([f"s{i % 7}" for i in range(70)], [str(i) for i in range(70)], "text")
    assert all(mask[i] == mask[i % 7] for i in range(70))


def test_config_file_and_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"count_max": 4}))
    assert load_config(str(p)).count_max == 4 and load_config(str(p), seed=3).seed == 3
    p.write_text(json.dumps({"count_mx": 4}))
    with pytest.raises(ValueError):
        load_config(str(p))


def test_build_cli_end_to_end(tmp_path):
    cats = [dict(id=1, name="person"), dict(id=18, name="dog")]
    images = [dict(id=i, width=W, height=H, file_name=f"COCO_train2014_{i:012d}.jpg") for i in (1, 2, 3)]

    def a(i, iid, cat, b):
        return dict(id=i, image_id=iid, category_id=cat, iscrowd=0, bbox=[b[0], b[1], b[2] - b[0], b[3] - b[1]])
    anns = [a(1, 1, 18, DOG_L), a(2, 2, 18, DOG_L), a(3, 2, 18, DOG_R), a(4, 3, 1, DOG_M)]
    (tmp_path / "coco.json").write_text(json.dumps(dict(categories=cats, images=images, annotations=anns)))
    (tmp_path / "eval.json").write_text(json.dumps({"bench": [99]}))
    (tmp_path / "mix.jsonl").write_text(json.dumps(rec(1, q("two dogs", "neg_count", False),
                                                       q("the dog", "cat_only", True, boxes=[DOG_L]))) + "\n")
    out = tmp_path / "out"
    subprocess.run([sys.executable, "-m", "glance_data.twins.build", "--mix", str(tmp_path / "mix.jsonl"), "--coco",
                    str(tmp_path / "coco.json"), "--eval-ids", str(tmp_path / "eval.json"), "--out", str(out),
                    "--config", "configs/twins_v14.json"], check=True, capture_output=True)
    recs = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    qs = {(r["image_id"], x["text"]): x for r in recs for x in r["questions"]}
    assert qs[(1, "two dogs")]["present"] is False and qs[(2, "two dogs")]["present"] is True
    assert qs[(3, "the dog")]["present"] is False and qs[(3, "the dog")]["twin_of"] == "cat_only"
    assert json.loads((out / "stats_twins.json").read_text())["count_pairs"] == 1
