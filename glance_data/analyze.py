"""Diagnose the current training data (JEV_PREP/train.jsonl and its sources) and the benchmarks.

python -m glance_data.analyze --out out/analysis.json
"""
import argparse
import collections
import json
import os
import random

import numpy as np

from glance_data.common import OUT, PREP, facets, head_nouns, load_grefs, load_instances, load_refs, mentioned_cats

rng = random.Random(0)


def q(v, ps=(10, 50, 90)):
    return {f"p{p}": float(np.percentile(v, p)) for p in ps} if len(v) else {}


def spec_level(f):
    if not f["noun"]:
        return "0_fragment"            # e.g. "left", "far right one", "white"
    k = f["spatial"] + f["appear"] + f["relation"]
    return ["1_category_only", "2_one_cue", "3_two_cues", "4_three_cues"][min(k, 3)]


def text_profile(texts, nouns_all):
    fs = [facets(t, nouns_all) for t in texts]
    lv = collections.Counter(spec_level(f) for f in fs)
    n = max(1, len(fs))
    return dict(n=len(fs), words=q([f["n"] for f in fs]), le2_words=sum(f["n"] <= 2 for f in fs) / n,
                ge12_words=sum(f["n"] >= 12 for f in fs) / n, levels={k: v / n for k, v in sorted(lv.items())},
                spatial=sum(f["spatial"] for f in fs) / n, appearance=sum(f["appear"] for f in fs) / n,
                relation=sum(f["relation"] for f in fs) / n, function=sum(f["function"] for f in fs) / n,
                spatial_only=sum(f["spatial"] and not f["appear"] and not f["relation"] for f in fs) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", default=PREP)
    ap.add_argument("--out", default=f"{OUT}/analysis.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    images, anns, cats = load_instances()
    nouns = head_nouns(cats)
    nouns_all = set().union(*nouns.values())
    refs = load_refs()
    grefs = load_grefs()
    train_imgs = {json.loads(line)["image_id"] for line in open(f"{a.prep}/train.jsonl")}
    R, ex = {}, collections.defaultdict(list)

    def inst(iid, min_frac=0.005):
        im = images[iid]; A = im["width"] * im["height"]
        return [o for o in anns[iid] if not o["crowd"] and o["area"] >= min_frac * A]

    # 1) specificity / length of the human expressions used for training, per source
    R["train_text"] = {}
    for ds in ["refcoco", "refcocoplus", "refcocog"]:
        texts = [s for r in refs if r["ds"] == ds and r["split"] == "train" and r["image_id"] in train_imgs for s in r["sents"]]
        R["train_text"][ds] = text_profile(texts, nouns_all)
        for t in rng.sample(texts, 400):
            f = facets(t, nouns_all)
            lv = spec_level(f)
            if len(ex[f"{ds}:{lv}"]) < 6:
                ex[f"{ds}:{lv}"].append(t)

    # 2) distractors: how many same-category instances does a training target have?
    tr = [r for r in refs if r["split"] == "train" and r["image_id"] in train_imgs]
    same = [sum(o["cat"] == r["cat"] for o in anns[r["image_id"]] if not o["crowd"]) for r in tr]
    R["targets_same_category_count"] = dict(n=len(same), eq1=float(np.mean([s == 1 for s in same])),
                                            ge2=float(np.mean([s >= 2 for s in same])),
                                            ge4=float(np.mean([s >= 4 for s in same])), dist=q(same, (10, 50, 90, 99)))
    # bare category queries: positives that are just the category name
    bare = [s for r in tr for s in r["sents"] if spec_level(facets(s, nouns_all)) == "1_category_only"]
    R["bare_category_positives"] = dict(n=len(bare), share=len(bare) / sum(len(r["sents"]) for r in tr),
                                        examples=rng.sample(bare, min(10, len(bare))))

    # 3) instance coverage on training images: which annotated objects ever get an expression?
    referred = {r["ann_id"] for r in refs}
    for g in grefs:
        referred |= {x for x in g["ann_id"] if x != -1}
    n_inst = n_ref = 0
    per_cat = collections.Counter(); per_cat_ref = collections.Counter()
    partial_imgs = 0; unref_examples = []
    for iid in train_imgs:
        objs = inst(iid)
        n_inst += len(objs); r_ = [o for o in objs if o["id"] in referred]; n_ref += len(r_)
        for o in objs:
            per_cat[cats[o["cat"]]] += 1; per_cat_ref[cats[o["cat"]]] += o["id"] in referred
        bycat = collections.defaultdict(list)
        for o in objs:
            bycat[o["cat"]].append(o["id"] in referred)
        if any(any(v) and not all(v) for v in bycat.values()):
            partial_imgs += 1
            if len(unref_examples) < 8:
                c = next(c for c, v in bycat.items() if any(v) and not all(v))
                unref_examples.append(dict(image_id=iid, category=cats[c], instances=len(bycat[c]), referred=sum(bycat[c])))
    R["instance_coverage"] = dict(instances=n_inst, referred=n_ref, referred_share=n_ref / n_inst,
                                  images_with_partially_referred_category=partial_imgs / len(train_imgs),
                                  least_covered=sorted([(c, per_cat_ref[c] / per_cat[c], per_cat[c]) for c in per_cat
                                                        if per_cat[c] >= 300], key=lambda x: x[1])[:12],
                                  examples=unref_examples)

    # 4) negatives actually used in training: are they hard (the named category IS in the image)?
    neg = collections.defaultdict(collections.Counter)
    for line in open(f"{a.prep}/train.jsonl"):
        r = json.loads(line)
        present_cats = {o["cat"] for o in anns[r["image_id"]]}
        for qq in r["questions"]:
            if qq["present"]:
                continue
            m = mentioned_cats(qq["text"], nouns)
            kind = "no_category_word" if not m else ("same_category_in_image" if m & present_cats else "category_absent")
            neg[qq["src"]][kind] += 1
            if len(ex[f"neg:{qq['src']}:{kind}"]) < 6 and rng.random() < 0.05:
                ex[f"neg:{qq['src']}:{kind}"].append(qq["text"])
    R["train_negatives"] = {s: dict(n=sum(c.values()), **{k: v / sum(c.values()) for k, v in c.items()}) for s, c in neg.items()}
    tot = collections.Counter()
    for c in neg.values():
        tot.update(c)
    R["train_negatives"]["all"] = dict(n=sum(tot.values()), **{k: v / sum(tot.values()) for k, v in tot.items()})

    # 5) gRefCOCO multi-target / no-target by split
    gs = collections.defaultdict(collections.Counter); ntd = collections.Counter()
    for g in grefs:
        k = "no_target" if g["no_target"] else ("multi" if len([x for x in g["ann_id"] if x != -1]) > 1 else "single")
        gs[g["split"]][k] += len(g["sentences"])
        if k == "multi":
            ntd[min(len(g["ann_id"]), 6)] += len(g["sentences"])
            if len(ex["multi"]) < 12 and rng.random() < 0.02:
                ex["multi"].append(f"{g['sentences'][0]['sent']}  (targets={len(g['ann_id'])})")
    R["grefcoco"] = dict(by_split={k: dict(v) for k, v in gs.items()}, multi_n_targets={str(k): v for k, v in sorted(ntd.items())})

    # 6) benchmarks
    B = {}
    for f in sorted(os.listdir(a.prep)):
        if not f.startswith("eval_") or "merged" in f:
            continue
        name = f[5:-6]
        rows = [json.loads(line) for line in open(f"{a.prep}/{f}")]
        qs = [qq for r in rows for qq in r["questions"]]
        texts = [qq["text"] for qq in qs if qq["present"]]
        b = dict(n_q=len(qs), n_img=len(rows), absent=sum(not qq["present"] for qq in qs) / len(qs),
                 multi=sum(qq.get("n_targets", 1) > 1 for qq in qs) / len(qs), text=text_profile(texts, nouns_all))
        person = 0
        for r in rows:
            for qq in r["questions"]:
                if qq["present"] and qq.get("box"):
                    m = mentioned_cats(qq["text"], nouns)
                    person += 1 in m
        b["mentions_person"] = person / max(1, len(texts))
        B[name] = b
    # same images reused across benchmarks
    img_sets = {n: {json.loads(line)["image_id"] for line in open(f"{a.prep}/eval_{n}.jsonl")} for n in B}
    B["_image_overlap"] = {f"{x}&{y}": len(img_sets[x] & img_sets[y]) for x in img_sets for y in img_sets
                           if x < y and len(img_sets[x] & img_sets[y]) > 0}
    R["benchmarks"] = B
    # category balance of training targets
    cc = collections.Counter(cats[r["cat"]] for r in tr for _ in r["sents"])
    tot_ = sum(cc.values())
    R["train_category_share_top"] = [(c, v / tot_) for c, v in cc.most_common(10)]
    R["train_categories_covered"] = len(cc)
    R["examples"] = ex
    json.dump(R, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in R.items() if k != "examples"}, indent=1, ensure_ascii=False)[:12000])


if __name__ == "__main__":
    main()
