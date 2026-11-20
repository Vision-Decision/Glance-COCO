"""Text-only probe gate: can the question text alone predict `present`?

A bag-of-n-grams logistic regression (TF-IDF over words 1-2 and characters 2-4) is trained on the question strings of
a training mixture, without the image, and scored by AUROC on held-out strings. 0.5 means the text carries no presence
signal. On a family whose every string is exactly twinned, a string-grouped split gives 0.5 by construction: the gate
then checks that the twins were built and nothing was left unpaired, not that a model needs the image. The gates (see TwinConfig.gates) are measured on subsets of the held-out questions:

  S1 rule      kinds that can be twinned by rule (TwinConfig.rule_kinds)       <= 0.65
  S2 count     count questions and their pairs                                  <= 0.55
  S3 position  position questions and their pairs                               <= 0.55
  S4 zh_rule   Chinese questions of the rule kinds                              <= 0.55
  S5 near      near-category negatives ("a toaster") and their pairs            <= 0.55
  S1_full, S4_full: the same on every box question / every Chinese question. Reported, not gated: LLM-written hard
  negatives cannot be twinned by rule, so some text signal is expected to remain there.

The held-out unit is the normalised STRING by default: every occurrence of a string lands on the same side. With an
image split, the two halves of an exact twin usually land on different sides; a probe that has memorised the train
twin's label then scores the test twin with the opposite label, and AUROC falls BELOW 0.5. That is an artefact of
exact pairing, not a reversed text signal. `--split image` is kept so the artefact can be shown.

The probe needs scikit-learn: pip install -e ".[probe]". Everything else here is numpy.

python -m glance_data.twins.gate --mix out/twins/train.jsonl --out out/twins/gate_text.json [--split image]
"""
import argparse
import json
import random

import numpy as np

from glance_data.twins.config import load_config


def question_kind(q):
    """Twin negatives count under the kind they twin, so that a family's subset holds both halves of each pair."""
    k = q.get("kind") or q.get("qtype") or ""
    return (q.get("twin_of") or k) if k == "mp_twin_neg" else k


def rows(records):
    """-> parallel lists (text, present, kind, lang, image id) over the box questions of `records`."""
    X, y, kind, lang, img = [], [], [], [], []
    for r in records:
        iid = r.get("image_id", r.get("img", r.get("file")))
        for q in r.get("questions", []):
            if q.get("type", "box") != "box" or not q.get("text"):
                continue
            X.append(q["text"]); y.append(int(bool(q["present"]))); kind.append(question_kind(q))
            lang.append(q.get("lang") or ""); img.append(str(iid))
    return X, np.array(y, dtype=int), kind, lang, img


def held_out(texts, images, split="text", frac=0.1, seed=0):
    """Boolean test mask. split="text": unit = lower-cased stripped string; split="image": unit = image id."""
    unit = [t.strip().lower() for t in texts] if split == "text" else list(images)
    units = sorted(set(unit))
    random.Random(seed).shuffle(units)
    test = set(units[: int(round(len(units) * frac, 9))])
    return np.array([u in test for u in unit], dtype=bool)


def auroc(y, s):
    """Mann-Whitney AUROC with tied scores averaged; None when only one class is present."""
    y, s = np.asarray(y, dtype=bool), np.asarray(s, dtype=float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2          # 1-based mean rank of each distinct score
    ranks = np.empty(len(s)); ranks[order] = avg_rank[inv]
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def subset_masks(kind, lang, cfg):
    k, zh = np.array(kind, dtype=object), np.array([x == "zh" for x in lang], dtype=bool)
    rule = np.isin(k, list(cfg.rule_kinds))
    return dict(all=np.ones(len(k), dtype=bool), count=np.isin(k, list(cfg.count_kinds)),
                position=np.isin(k, list(cfg.position_kinds)), near=np.isin(k, list(cfg.near_gate_kinds)), zh=zh, rule=rule, zh_rule=rule & zh)


def score_gates(y, s, kind, lang, cfg):
    """y, s, kind, lang over the held-out questions -> {gate: dict(subset, n, present_rate, auroc, threshold, passed)}."""
    masks, out = subset_masks(kind, lang, cfg), {}
    for g, (sub, thr) in cfg.gates.items():
        m = masks[sub]
        auc = auroc(y[m], s[m])
        out[g] = dict(subset=sub, n=int(m.sum()), present_rate=float(y[m].mean()) if m.any() else None, auroc=auc,
                      threshold=thr, passed=None if auc is None else auc <= thr)
    return out


def fit_probe(texts, y):
    """The text-only probe; returns a function texts -> P(present)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    clf = make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=400000),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=5, sublinear_tf=True, max_features=400000)),
        LogisticRegression(C=2.0, max_iter=2000))
    clf.fit(texts, y)
    return lambda t: clf.predict_proba(t)[:, 1]


def run(records, cfg, split="text", probe=fit_probe):
    X, y, kind, lang, img = rows(records)
    if len(X) > cfg.probe_cap:
        keep = sorted(random.Random(0).sample(range(len(X)), cfg.probe_cap))
        X, y = [X[i] for i in keep], y[keep]
        kind, lang, img = [kind[i] for i in keep], [lang[i] for i in keep], [img[i] for i in keep]
    te = held_out(X, img, split, cfg.test_frac)
    tr, ti = np.where(~te)[0], np.where(te)[0]
    score = probe([X[i] for i in tr], y[tr])
    s = np.asarray(score([X[i] for i in ti]), dtype=float)
    yt, kt, lt = y[ti], [kind[i] for i in ti], [lang[i] for i in ti]
    res = dict(split=split, n_questions=len(X), n_test=int(te.sum()), present_rate=float(y.mean()),
               gates=score_gates(yt, s, kt, lt, cfg))
    # where the remaining signal sits: mean P(present) per kind on the held-out questions
    per = {}
    for k in sorted(set(kt)):
        m = np.array([x == k for x in kt])
        if m.sum() >= 200:
            per[k] = dict(n=int(m.sum()), present_rate=float(yt[m].mean()), mean_p=float(s[m].mean()))
    res["per_kind"] = per
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="text", choices=["text", "image"],
                    help="held-out unit; text keeps both halves of a twin on the same side (see the module docstring)")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    res = run((json.loads(line) for line in open(a.mix)), cfg, a.split)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["gates"], indent=1))


if __name__ == "__main__":
    main()
