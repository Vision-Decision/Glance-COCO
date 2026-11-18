"""Export assembled records (aug_*.jsonl) to Parquet shards for a Hugging Face dataset. No images are exported:
rows reference COCO train2014 images by `image_id` / `file`.

python -m glance_data.export_hf --src out/aug_train_all.jsonl --split train --out hf/data
python -m glance_data.export_hf --src out/aug_refcoco_val.jsonl --split diagnostic --out hf/data

Row = one image: image_id, file, width, height, questions (list of struct: text, lang, kind, src, present,
n_targets, boxes). `boxes` holds every target as [x0, y0, x1, y1] in pixels; it is empty when present is false.
"""
import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

QUESTION = pa.struct([
    ("text", pa.string()), ("lang", pa.string()), ("kind", pa.string()), ("src", pa.string()),
    ("present", pa.bool_()), ("n_targets", pa.int32()), ("boxes", pa.list_(pa.list_(pa.float32()))),
])
SCHEMA = pa.schema([("image_id", pa.int64()), ("file", pa.string()), ("width", pa.int32()), ("height", pa.int32()),
                    ("questions", pa.list_(QUESTION))])


def to_row(r):
    qs = [dict(text=q["text"], lang=q.get("lang", "en"), kind=q["kind"], src=q["src"], present=bool(q["present"]),
               n_targets=int(q["n_targets"]), boxes=[[round(float(v), 2) for v in b] for b in (q.get("boxes") or [])])
          for q in r["questions"]]
    return dict(image_id=r["image_id"], file=r["file"], width=r["W"], height=r["H"], questions=qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows-per-shard", type=int, default=6000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [to_row(json.loads(line)) for line in open(a.src)]
    rows.sort(key=lambda r: r["image_id"])
    n = (len(rows) + a.rows_per_shard - 1) // a.rows_per_shard
    nq = 0
    for i in range(n):
        part = rows[i * a.rows_per_shard:(i + 1) * a.rows_per_shard]
        nq += sum(len(r["questions"]) for r in part)
        pq.write_table(pa.Table.from_pylist(part, schema=SCHEMA), f"{a.out}/{a.split}-{i:05d}-of-{n:05d}.parquet",
                       compression="zstd")
    print(f"{a.split}: {len(rows)} images, {nq} questions, {n} shards -> {a.out}")


if __name__ == "__main__":
    main()
