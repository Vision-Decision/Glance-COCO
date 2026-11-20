# Minimal pairs: negatives that text alone cannot give away

`glance_data.twins` gives rule-made questions a **twin**: the same string, on a different image, with the opposite
answer. After this step a model cannot tell whether a rule-made question is answered "present" or "not present" from
its wording. It has to look at the image.

## The problem: the template gives the answer away

Rule-made negatives are created *because* the image contradicts them. `assemble.code_items` writes "two clocks" only
when the image does not have two clocks, and "the cup on the left" only when every cup is on the right. In a training
mix, "number + noun" questions are therefore almost always absent, and "the X" questions are almost always present.
A model can learn presence from the string and never look at the image.

We measured this with a **text-only probe**: a logistic regression on word and character n-grams of the question text
that never sees the image. It predicts `present` and is scored by AUROC on held-out strings, where 0.5 means the text
carries no signal. On the v1.3 training mix the probe reaches 0.948 on count questions and 0.959 on Chinese questions
of the twinnable kinds, so those could be answered without the image.

## The fix: one string, both answers

Every new question gets its truth from COCO annotations, as in `assemble.py`. No LLM is involved.

| Builder | Source question | Twin | Image rule |
|---|---|---|---|
| Count pairs | `neg_count` "two clocks" / "两个钟" (absent) | same string, **present** (`mp_count_pos`) | exactly that many instances, 2 ≤ n ≤ 5, each reliable; never a pile-prone category (book, banana, carrot, broccoli, apple, orange, donut) |
| Position pairs | `neg_position` "the cup on the left" / "左边的杯子" (absent) | same string, **present** (`mp_pos_pos`) | the image's only cup is reliable and its centre is in the left (right) third |
| Missing-category twins | a present question of a rule-answered kind: `cat_only`, `cat_all`, `cf_anchor`, `pos_pos_sole`, `count_pos` (and the category-free `cf_func`, `cf_generic`, `cf_rel`, whose target is a known COCO box) | same string, **absent** (`mp_twin_neg`, `twin_of` = source kind) | no instance of any category the string names or the target belongs to. For `count_pos` ("the two dogs") an image where the category is present **with a different count** is preferred, which is the harder negative |
| Chinese pairs | a Chinese RefCOCO translation, `ref_zh` (present) | same string, **absent** (`mp_zh_neg`) | no instance of the target's category or of any category whose Chinese name occurs in the string |

"Reliable" means: not crowd, at least 0.5% of the image, and not a sliver under 2% of the image cut off at the border.
A target's category is found from the text (en/zh names and synonyms in `twins/lexicon.py`), from `ref_cat` when the
record has it, and from the source-image annotation that coincides with the target box.

**Lone negatives are dropped.** If no eligible image exists for a count or position string ("six dogs"; "the toaster
on the right"), the negative is removed instead of being kept unpaired. Unpaired negatives would bring the shortcut
back. In v1.4 this removed 4,137 count negatives.

Every twin image is outside the held-out evaluation images (`--eval-ids`) and is never the source image. Images are
drawn round-robin from a seeded shuffle, so pairs spread over many images. Given `--seed` and the input order, the
output is deterministic.

## v1.4 build

Input: the v1.3 training mix. Candidate images: COCO train2014 minus every evaluation image. Re-running
`glance_data.twins.build` with `configs/twins_v14.json` on the same inputs reproduces the published v1.4 questions
exactly (335,139 new questions, identical image, text, answer and boxes).

| | Questions |
|---|---:|
| Count pairs (`mp_count_pos`) | 33,617 |
| Count negatives dropped (no eligible image) | 4,137 |
| Position pairs (`mp_pos_pos`) | 11,806 |
| Chinese pairs (`mp_zh_neg`) | 57,542 |
| Missing-category twins (`mp_twin_neg`) | 218,002 |
| ... of which `count_pos` twins on a wrong-count image | 14,172 |

The mix itself is in the private Hugging Face dataset `Vision-Decision/glance-train-data`, folder `v1.4/`
(`train.jsonl`, `stats_mp.json`, `gate_text.json`, `gate_image.json`). The data is not in this repository. The build
statistics and gate summaries are in `docs/results/twins_*.json`.

## The gate

`python -m glance_data.twins.gate` trains the probe on 90% of the strings of a mix and reports AUROC on the remaining
10%:

| Gate | Subset | Max | v1.3 mix | v1.4 mix |
|---|---|---:|---:|---:|
| S1 | rule-twinnable kinds | 0.65 | 0.954 | **0.500** |
| S2 | count questions | 0.55 | 0.948 | **0.500** |
| S3 | position questions | 0.55 | 0.753 | **0.500** |
| S4 | Chinese, rule-twinnable kinds | 0.55 | 0.959 | **0.500** |
| S1_full | every box question (reported) | – | 0.853 | 0.722 |
| S4_full | every Chinese question (reported) | – | 0.942 | 0.594 |

All four gates pass, and on every rule-twinnable subset the text-only probe is exactly at chance.

**What remains is the LLM-written hard negatives.** S1_full and S4_full still sit above chance because of
`hneg_attribute` / `hneg_relation` and other LLM-written negatives ("a black cow" when every cow is brown). They
cannot be twinned by rule. A twin would need an image on which that exact sentence is true, and annotations cannot
decide that. These negatives are what teaches attribute and relation checking, so they are kept. The gate therefore
scores the rule-twinnable kinds and reports the full pool next to them.

### Split by string, not by image

The held-out unit is the normalised string (`--split text`, the default), so every copy of a string lands on the
same side. With `--split image`, the two halves of an exact twin usually land on different sides. The probe has then
seen the training twin with the *opposite* label, scores the test twin with that label, and AUROC drops below 0.5:

| v1.4 mix | by string | by image |
|---|---:|---:|
| S1 (rule kinds) | 0.500 | 0.397 |
| S2 count | 0.500 | 0.373 |
| S3 position | 0.500 | 0.393 |
| S4 Chinese rule kinds | 0.500 | 0.339 |

Values below 0.5 are an artefact of exact pairing, not a reversed text signal. `tests/test_twins.py` reproduces this
with a probe that only memorises training strings: 0.5 by string, close to 0 by image.

## Run it

```bash
pip install -e ".[probe]"                 # the gate needs scikit-learn; the builder does not
make twins MIX=out/aug_train.jsonl COCO_ANN=/path/to/instances_train2014.json EVAL_IDS=/path/to/eval_image_ids.json
make twin-gate                            # out/twins/gate_text.json; add SPLIT_BY=image for the artefact check
```

The builder reads `kind` (Glance-Data records) or `qtype` (prepared training mixes). It copies the input records,
minus dropped negatives, and appends the new records. All settings, and the defaults used for v1.4, are in
`configs/twins_v14.json`.

## Limitations

- **LLM-written hard negatives have no twin** (see above). The full-pool AUROC stays above chance by construction.
- **COCO misses objects.** An absent twin can be wrong when the category is in the image but not annotated. This is
  the same risk as the existing `neg_near` negatives.
- **Category matching is lexical.** A Chinese sentence whose noun is not in the lexicon, and whose box matches no
  annotation, gets no twin (counted as `*_no_category` in the stats). Its category cannot be identified, so it is not
  safe to twin.
