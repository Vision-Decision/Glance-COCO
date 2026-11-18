"""Before/after statistics: the original training file vs. original + augmentation.

python -m glance_data.compare --orig $JEV_PREP/train.jsonl --aug out/aug_train_all.jsonl --out out/compare.json
"""
import argparse
import collections
import json

from glance_data.analyze import spec_level
from glance_data.common import facets, head_nouns, load_instances, mentioned_cats

HARD_KINDS = {"hneg_attribute", "hneg_relation", "neg_count", "neg_position"}   # named category IS in the image


def profile(path, nouns, nouns_all, anns, augmented):
    c = collections.Counter()
    levels = collections.Counter()
    for line in open(path):
        r = json.loads(line)
        present = {o["cat"] for o in anns[r["image_id"]]}
        for q in r["questions"]:
            c["questions"] += 1
            lang = q.get("lang", "en")
            c[f"lang_{lang}"] += 1
            if q["present"]:
                c["positives"] += 1
                c["multi_target"] += q.get("n_targets", 1) > 1
                if lang == "en":
                    levels[spec_level(facets(q["text"], nouns_all))] += 1
                if q.get("kind") == "cat_only":
                    c["bare_category_single"] += 1
                if q.get("kind") == "function":
                    c["use_or_role"] += 1
            else:
                c["negatives"] += 1
                if augmented:
                    hard = q.get("kind") in HARD_KINDS
                else:
                    hard = bool(mentioned_cats(q["text"], nouns) & present)
                c["neg_same_category_in_image"] += hard
    n_en = sum(levels.values())
    return dict(counts=dict(c), en_positive_specificity={k: v / n_en for k, v in sorted(levels.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--aug", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _, anns, cats = load_instances()
    nouns = head_nouns(cats)
    nouns_all = set().union(*nouns.values())
    o = profile(a.orig, nouns, nouns_all, anns, False)
    g = profile(a.aug, nouns, nouns_all, anns, True)
    tot = collections.Counter(o["counts"]) + collections.Counter(g["counts"])
    res = dict(original=o, augmentation=g, combined_counts=dict(tot))
    json.dump(res, open(a.out, "w"), indent=1)
    for name, c in [("original", o["counts"]), ("augmentation", g["counts"]), ("combined", dict(tot))]:
        print(name, json.dumps(c))
    print("en specificity  original", {k: round(v, 3) for k, v in o["en_positive_specificity"].items()})
    print("en specificity  aug     ", {k: round(v, 3) for k, v in g["en_positive_specificity"].items()})


if __name__ == "__main__":
    main()
