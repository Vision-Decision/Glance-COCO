---
license: cc-by-nc-4.0
language:
- en
- zh
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
pretty_name: Glance-Data
size_categories:
- 1M<n<10M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: diagnostic
    path: data/diagnostic-*.parquet
---

# Glance-Data

Grounding data for a Jev-style task. Each example is one image plus many text requests. Every request is answered in
parallel with one box, several boxes, or "not present".

This dataset augments the RefCOCO-family training data. It adds the kinds of requests that the original data lacks:
- a ladder of specificity (short / medium / long) and casual phrasing, all in English and Chinese;
- category-only requests ("the tie") and all-instance requests ("all the zebras");
- multi-target requests and descriptions by use ("something to sit on");
- hard negatives: the same category is present, but a detail is wrong ("a black cow" when every cow is white-brown);
  count, position and near-category negatives.

**Ground truth comes from COCO annotations. An LLM only writes the wording.** Boxes, counts and "is this category in
the image" are computed from the annotations. DeepSeek (deepseek-flash, vision) looked at each image and wrote the
sentences for numbered objects. Rule filters remove sentences that leak the drawing and "attribute" negatives that
changed the position instead.

## Splits

| split | images | questions | images from |
|---|---:|---:|---|
| `train` | 24,406 | 1,831,146 | RefCOCO-family training images, with every benchmark image removed |
| `diagnostic` | 300 | 23,578 | `refcoco_val` images, never used for training |

No image appears in both splits. No `train` image appears in any RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO evaluation
set (6,549 images held out).

## Fields

One row per image:

| field | type | meaning |
|---|---|---|
| `image_id` | int | COCO image id |
| `file` | string | `COCO_train2014_<id>.jpg` (COCO 2014 train images; not included here) |
| `width`, `height` | int | original image size |
| `questions` | list | requests for this image |

Each question:

| field | meaning |
|---|---|
| `text` | the request |
| `lang` | `en` / `zh` |
| `kind` | see below |
| `src` | `aug_llm` (LLM wording), `aug_function` (LLM names the categories, code picks every instance), `aug_code` (fully rule-made), `aug_gref` (human gRefCOCO multi-target sentence) |
| `present` | whether the request matches something in the image |
| `n_targets` | 0, 1 or more |
| `boxes` | every target as `[x0, y0, x1, y1]` in pixels of the original image; empty when `present` is false |

| kind | answer | ground truth |
|---|---|---|
| `short`, `medium`, `long`, `casual` | 1 box | the object the LLM was asked to describe |
| `group` | ≥2 boxes | the objects the LLM was asked to describe together |
| `function` | ≥1 box | every instance of the COCO categories the LLM named for that use |
| `cat_only` / `cat_all` | 1 / all boxes | COCO annotations (category with one instance / all instances) |
| `gref_multi` | ≥2 boxes | human gRefCOCO annotation |
| `hneg_attribute`, `hneg_relation` | not present | LLM edit of one attribute or action, position words kept identical |
| `neg_count`, `neg_position`, `neg_near` | not present | COCO annotations (more than present / no instance on that side / absent near category) |

## Quality

A stratified human audit judged 8 random items per kind, 112 in total: **101/112 (≈90%) correct**. Every kind scored
≥6/8 except category-only (5/8), which was then fixed by skipping slivers under 0.5% of the image. Hard negatives
scored 16/16. The most common remaining error is a short description that is not unique (for example "green tie"
when there are two green ties).

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Vision-Decision/Glance-Data")
row = ds["train"][0]
for q in row["questions"][:5]:
    print(q["kind"], q["lang"], q["present"], q["text"], q["boxes"])
```

Images come from COCO 2014 train (`file`). A model with a single-box head can use one target per question. Using
`group`, `cat_all`, `function` and `gref_multi` fully needs a multi-box head.

## License and sources

- Images: COCO (not redistributed here); COCO annotations are CC BY 4.0.
- Referring expressions: RefCOCO / RefCOCO+ / RefCOCOg (research use) and gRefCOCO (**non-commercial**). Because of
  gRefCOCO, the dataset as a whole is released under CC BY-NC 4.0.
- Generated text: written by DeepSeek (deepseek-flash) from the images.

Pipeline code: `Vision-Decision/Glance-Data` on GitHub.
