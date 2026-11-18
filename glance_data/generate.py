"""Stage 1 (runs where the API key is; needs only the inventory folder): one DeepSeek vision call per image.

  DEEPSEEK_API_KEY_FILE=/path/to/key python -m glance_data.generate --inv INV_DIR --out OUT_DIR [--limit N] [--workers 16]

Input per image: the clean photo, the same photo with numbered boxes on <=6 objects, and JSON facts (category,
box and existing human descriptions of each marked object; counts of every annotated category in the photo).
Output per marked object: a specificity ladder (short / medium / long), a casual phrasing, Chinese versions and
minimally-edited hard negatives; per image: group expressions (2+ objects) and use/role descriptions whose
targets are decided later by code from the COCO categories the model names.
Output: OUT_DIR/gen.jsonl, one line per image (resumable). Thinking is disabled (JSON only, ~0.005 CNY / image).
"""
import argparse
import base64
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

API = "https://api.deepseek.com/chat/completions"

GEN_SYSTEM = """You write training data for a visual grounding model. The model gets a photo and a text query and must decide whether the query matches something in the photo and, if so, box it.

You get two images: image 1 is the clean photo, image 2 is the same photo with numbered coloured boxes ("marks") on some objects. You also get JSON facts: for each mark its COCO category, its box (x0,y0,x1,y1 as fractions of width/height) and existing human descriptions (may be empty, terse or noisy), plus the count of every annotated COCO category in the whole photo (marked and unmarked objects).

Hard rules
- Write like a normal person describing the photo. Never mention boxes, marks, numbers of marks, outline colours, "image 1/2", "annotated" or "highlighted".
- A positive description must match exactly the intended object(s) and nothing else in the photo. Check ALL objects in the photo, marked or not, including ones the counts tell you exist, and objects of OTHER categories that the same words could cover (e.g. "controller" covers both a remote and a nunchuk; "food" covers every dish; "vehicle" covers cars, buses and bikes).
- Use only what is clearly visible. Do not invent identities, brands, unreadable text, gender or age you cannot see clearly.
- English must be natural. Chinese must be natural Simplified Chinese written for a Chinese user, not a word-by-word translation.

For each marked object (set "skip": true if it is too small, cut off or occluded to describe uniquely):
1 "short": 1-4 words, the minimal unique description (e.g. "left zebra", "red umbrella"). If the plain category word is already unique in the photo, use just that.
2 "medium": 4-9 words with one appearance cue and one location or relation cue.
3 "long": 14-30 words, a detailed description with several visible attributes (clothing, colour, pose, action, relation), as a careful annotator would write.
4 "casual": an informal or indirect way to refer to it (synonym, hypernym, slang, role, e.g. "the pup in the orange pack", "the guy behind the counter"); still unique - add a cue if the casual word alone fits other objects.
5 "zh_short", "zh_medium", "zh_long", "zh_casual": Chinese versions of 1-4.
6 "hard_negatives": 2 queries made by minimally editing the medium description so that it matches NOTHING in the photo: "attribute" (change one clearly visible attribute, such as a colour, to a contrasting value that NO object the words could cover has; e.g. black cat -> orange cat), "relation" (change the action or relation to one that clearly does not happen). Do not make position edits. Give "text", "zh", "edit".

For the whole photo:
7 "groups": 1-3 expressions that refer to two or more marked objects together (e.g. "the two people at the table", "the man in red and the bowl in front of him"), each with "marks", "text", "zh". The text must name every one of those objects (not just one of them with the others as context), must match exactly those objects and no other object in the photo; skip if impossible.
8 "functions": 1-3 descriptions by use or role (e.g. "something to sit on", "what you'd drink from", "food"), each with "text", "zh" and "categories": the COCO categories from the counts whose objects serve that use in this photo (all of them, e.g. ["chair", "couch", "bed"]). Only use a function if every visible object that serves it belongs to one of the listed COCO categories.

Reply with JSON only:
{"objects":[{"mark":1,"skip":false,"short":"","medium":"","long":"","casual":"","zh_short":"","zh_medium":"","zh_long":"","zh_casual":"","hard_negatives":[{"text":"","zh":"","edit":"attribute"}]}],"groups":[{"marks":[1,2],"text":"","zh":""}],"functions":[{"text":"","zh":"","categories":["chair"]}]}"""

# words that give away the set-of-mark drawing; such queries are dropped
MARK_LEAK = re.compile(r"\b(box|boxes|boxed|mark|marked|label(l)?ed|outlined|highlighted|number \d|#\d|image [12])\b|框|标记|编号", re.I)


def b64(path):
    return "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()


class Client:
    def __init__(self, key, model="deepseek-flash", max_tokens=6000):
        self.key, self.model, self.max_tokens = key, model, max_tokens
        self.lock = threading.Lock()
        self.usage = dict(prompt=0, completion=0, calls=0, errors=0)

    def __call__(self, system, images, text, tries=4):
        content = [{"type": "image_url", "image_url": {"url": b64(p)}} for p in images] + [{"type": "text", "text": text}]
        body = dict(model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                    response_format={"type": "json_object"}, max_tokens=self.max_tokens, temperature=0.7,
                    thinking={"type": "disabled"})
        for t in range(tries):
            try:
                req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={
                    "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                r = json.loads(urllib.request.urlopen(req, timeout=300).read())
                u = r.get("usage", {})
                with self.lock:
                    self.usage["prompt"] += u.get("prompt_tokens", 0)
                    self.usage["completion"] += u.get("completion_tokens", 0)
                    self.usage["calls"] += 1
                return json.loads(r["choices"][0]["message"]["content"])
            except Exception:  # network / rate limit / bad JSON
                with self.lock:
                    self.usage["errors"] += 1
                if t == tries - 1:
                    raise
                time.sleep(2 + 4 * t)


def facts(rec):
    return json.dumps(dict(counts=rec["counts"], marks=[dict(mark=m["mark"], category=m["cat"], box=m["box_norm"],
                                                             human_descriptions=m["human"]) for m in rec["marked"]]))


def collect_queries(rec, gen):
    """-> list of dict(text, lang, kind, intent): intent = [marks] (positive), "none" (negative) or "code"
    (function: targets = all instances of q["categories"], decided in assemble.py)."""
    qs = []
    valid = {m["mark"] for m in rec["marked"]}
    for o in gen.get("objects", []):
        mk = o.get("mark")
        if mk not in valid or o.get("skip"):
            continue
        for k, lang in [("short", "en"), ("medium", "en"), ("long", "en"), ("casual", "en"),
                        ("zh_short", "zh"), ("zh_medium", "zh"), ("zh_long", "zh"), ("zh_casual", "zh")]:
            t = o.get(k)
            if isinstance(t, str) and t.strip():
                qs.append(dict(text=t.strip(), lang=lang, kind=k.replace("zh_", ""), intent=[mk]))
        for h in o.get("hard_negatives") or []:
            for k, lang in [("text", "en"), ("zh", "zh")]:
                t = h.get(k)
                if isinstance(t, str) and t.strip():
                    qs.append(dict(text=t.strip(), lang=lang, kind=f"hneg_{h.get('edit', 'attribute')}", intent="none", mark=mk))
    for g in gen.get("groups") or []:
        mks = sorted({m for m in g.get("marks", []) if m in valid})
        if len(mks) < 2:
            continue
        for k, lang in [("text", "en"), ("zh", "zh")]:
            t = g.get(k)
            if isinstance(t, str) and t.strip():
                qs.append(dict(text=t.strip(), lang=lang, kind="group", intent=mks))
    for fn in gen.get("functions") or []:
        cats = [c for c in fn.get("categories") or [] if c in rec["counts"]]
        for k, lang in [("text", "en"), ("zh", "zh")]:
            t = fn.get(k)
            if isinstance(t, str) and t.strip() and cats:
                qs.append(dict(text=t.strip(), lang=lang, kind="function", intent="code", categories=cats))
    seen, out = set(), []
    for q in qs:
        if q["text"].lower() in seen:
            continue
        seen.add(q["text"].lower())
        q["leak"] = bool(MARK_LEAK.search(q["text"]))
        out.append(q)
    return out


def process(rec, inv, client):
    imgs = [f"{inv}/img/{rec['image_id']}_clean.jpg", f"{inv}/img/{rec['image_id']}_marked.jpg"]
    gen = client(GEN_SYSTEM, imgs, "Facts:\n" + facts(rec))
    return dict(image_id=rec["image_id"], gen=gen, queries=collect_queries(rec, gen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    client = Client(open(os.environ["DEEPSEEK_API_KEY_FILE"]).read().strip())
    os.makedirs(a.out, exist_ok=True)
    recs = [json.loads(line) for line in open(f"{a.inv}/inventory.jsonl")]
    if a.limit:
        recs = recs[: a.limit]
    path = f"{a.out}/gen.jsonl"
    done = {json.loads(line)["image_id"] for line in open(path)} if os.path.exists(path) else set()
    todo = [r for r in recs if r["image_id"] not in done]
    t0, n_ok, n_fail = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex, open(path, "a") as f:
        futs = {ex.submit(process, r, a.inv, client): r for r in todo}
        for fu in as_completed(futs):
            try:
                f.write(json.dumps(fu.result(), ensure_ascii=False) + "\n"); f.flush()
                n_ok += 1
            except Exception as e:
                n_fail += 1
                print("fail", futs[fu]["image_id"], repr(e)[:200], flush=True)
            if (n_ok + n_fail) % 100 == 0:
                print(f"{n_ok + n_fail}/{len(todo)} ok {n_ok} fail {n_fail} {time.time() - t0:.0f}s {client.usage}", flush=True)
    print(f"done ok {n_ok} fail {n_fail} {time.time() - t0:.0f}s {client.usage}", flush=True)
    json.dump(client.usage, open(f"{a.out}/usage_{int(time.time())}.json", "w"))


if __name__ == "__main__":
    main()
