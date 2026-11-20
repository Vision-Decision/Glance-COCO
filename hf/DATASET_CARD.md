---
license: cc-by-nc-sa-4.0
language:
- en
- zh
pretty_name: Glance-COCO
task_categories:
- object-detection
- zero-shot-object-detection
tags:
- visual-grounding
- referring-expression-comprehension
- refcoco
- coco
- hard-negatives
- multi-target
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: diagnostic
    path: data/diagnostic-*.parquet
- config_name: plus
  data_files:
  - split: train
    path: plus/train-*.parquet
- config_name: diagnostic_pos_sole
  data_files:
  - split: test
    path: eval/parquet/diagnostic_pos_sole.parquet
- config_name: robust
  data_files:
  - split: zh
    path: eval/parquet/robust-zh.parquet
  - split: para
    path: eval/parquet/robust-para.parquet
  - split: func
    path: eval/parquet/robust-func.parquet
  - split: vague
    path: eval/parquet/robust-vague.parquet
---

# Glance-COCO

Grounding data on COCO images: one image, many text requests, and every request is answered with one box, several
boxes, or "not present". This repository was called **Glance-Data**; the old URL redirects here, so
`load_dataset("Vision-Decision/Glance-Data")` keeps working.

It now also holds the COCO-image datasets that were published separately: the training supplements that fix
the main data's known shortcuts, the trainer-format records, the evaluation sets and the build intermediates. Each one
is a config or a folder here, and its original card is kept unchanged in [`provenance/`](provenance/).

**Ground truth comes from COCO annotations. Language models only write the wording.** Images are COCO 2014 train and
are not included.

## Configs

| config | split(s) | rows | use | formerly |
|---|---|---:|---|---|
| `default` | `train`, `diagnostic` | 24,406 / 300 images | the main data: 1,831,146 / 23,578 requests in 14 kinds | Glance-Data |
| `plus` | `train` | 57,092 rows, 317,500 requests | **train-only supplements** that fix shortcuts in `default` (see below); same schema plus a `source` column, so it concatenates with `default` | Glance-Data-Plus-clean |
| `diagnostic_pos_sole` | `test` | 160 records, 764 requests | **evaluation only**: sole-instance position / superlative positives on the diagnostic images | glance-ground-v2 |
| `robust` | `zh`, `para`, `func`, `vague` | 6,236 items | **evaluation only**: paraphrased, functional, vague and Chinese requests on RefCOCO/gRefCOCO val images | glance-robust-ref-clean |

## Trainer-format records (jsonl files)

`records/` holds `default` + `pos_pos_sole` in the Glance trainer's record format, as plain jsonl (formerly
glance-ground-v2): `full/` (51,177 records, ≤48 requests each, every target in `boxes`, for multi-box heads),
`single/` (45,027 records, `n_targets` ≤ 1, for single-box heads) and `diagnostic.jsonl` (300 records). They are files,
not configs, because one Hub dataset can only expose one file format:

```python
from datasets import load_dataset
full = load_dataset("json", data_files="hf://datasets/Vision-Decision/Glance-COCO/records/full/*.jsonl", split="train")
```

## What `plus` fixes

| problem in `default` | fix | source in `plus` |
|---|---|---|
| position words ("the leftmost cat") appear only in negatives, so a model learns to reject them (Glance v1 rejected 86.5% of sole-instance position requests) | rule-made positives on the only instance of a category, sampled 1:1 against `neg_position` | `ground_v2_gap_rule` |
| `neg_count` can be recognised from the text alone 98.6% of the time | "exactly n" positives (n = 2–5), so a count becomes a minimal pair | `v1.3_count_pairs` |
| almost every request names the category | function, supercategory and relation requests without the category word, with negatives of the same form; blind re-identification check plus a leak lexicon | `v1.2_catfree` |
| regressions seen in v1.2 | targeted negatives: absent class at a position or in a relation | `v1.3_negatives` |

`plus/manifest/` lists, by SHA-1 only, the RefCOCO expressions that were replayed during training and the Chinese
translations that are not redistributed (licence of RefCOCO-family text is unclear).

## Folders

| folder | contents |
|---|---|
| `data/` | `default` parquet files (unchanged paths) |
| `plus/` | `plus` parquet files, `manifest/`, `stats.json` |
| `records/` | trainer-format jsonl: `full/`, `single/`, `diagnostic.jsonl`, `stats.json` |
| `eval/` | the original `diagnostic_pos_sole.jsonl` and `robust/` jsonl files (with the `withheld/` reference list and `stats.json`); `parquet/` holds row-for-row parquet copies that back the `diagnostic_pos_sole` and `robust` configs |
| `intermediates/` | build intermediates: rule-made scene specs, raw and filtered category-free generations (`catfree_v12/`), v1.3 negatives and count positives, decontamination id and hash lists, the Glance-X data preparation script and provenance |
| `provenance/` | the original cards of the five merged repositories, unchanged |

## Decontamination

All training images are COCO train2014. No training image appears in the RefCOCO/+/g val or test splits, the
gRefCOCO val/testA/testB splits, COCO val2017, the `diagnostic` split or the `robust` set. The `robust` and
`diagnostic*` configs are for evaluation only: do not train on them.

## Licence

CC BY-NC-SA 4.0, for non-commercial research only (inherited from gRefCOCO). Images are not included; COCO
annotations are CC BY 4.0. Per-part provenance and licences are in the original cards under `provenance/`.

Pipeline code: [Vision-Decision/Glance-COCO](https://github.com/Vision-Decision/Glance-COCO) on GitHub.
