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
| Near-category pairs (v1.5) | `neg_near` "a toaster" / "一台烤面包机" (absent) | same string, **present** (`mp_near_pos`) | the image's only toaster is reliable |

"Reliable" means: not crowd, at least 0.5% of the image, and not a sliver under 2% of the image cut off at the border.
A target's category is found from the text (en/zh names and synonyms in `twins/lexicon.py`), from `ref_cat` when the
record has it, and from the source-image annotation that coincides with the target box. A word that can name several
categories ("bag": backpack and handbag; "glass": cup and wine glass; "瓶": bottle and vase) rules out all of them. For
the category-free kinds (`cf_func`, `cf_generic`, `cf_rel`) the absent image also has no instance of the target's
COCO supercategory or of its confusable classes (`assemble.NEAR`): "something to sit on" written for a chair is just
as true of a couch or a bench.

**Lone negatives are dropped.** If no eligible image exists for a count or position string ("six dogs"; "the toaster
on the right"), the negative is removed instead of being kept unpaired. Unpaired negatives would bring the shortcut
back. In v1.4 this removed 4,137 count negatives. Since v1.5 the same holds for near-category negatives. Strings that
do not match a template are counted apart (`*_neg_unparsed`) from those that found no image (`*_neg_dropped`).

Every twin image is outside the held-out evaluation images (`--eval-ids`) and is never the source image. Images are
drawn round-robin from a seeded shuffle, so pairs spread over many images. Given `--seed` and the input order, the
output is deterministic.

## v1.5: label fixes and near-category pairs

v1.4 left two families of label errors and one unpaired family. v1.5 (`configs/twins_v15.json`, the default) fixes
them. It has not yet been rebuilt on the full mix: the numbers below are still those of v1.4.

| Problem in v1.4 | v1.5 |
|---|---|
| A `count_pos` twin ("the two dogs", absent) could land on an image whose dog count is uncertain: a crowd region, or two clear dogs plus an annotated speck (3 annotations) | the wrong-count image must have every instance reliable and no crowd; otherwise the twin falls back to an image without the category |
| An ambiguous word named one category only ("bag" -> backpack), so "the man with a bag" could be made absent on an image with a handbag | the word rules out every category it can name |
| A category-free description could be made absent on an image where another class fits it ("something to sit on" on a couch) | supercategory and confusable classes are ruled out too |
| `neg_near` ("a toaster") was never present anywhere: v1.4's probe gave it a mean presence of 0.215 | present twin on an image with one reliable instance (`mp_near_pos`), or the negative is dropped; gate S5 |
| Chinese rule strings used 个 for every noun ("两个狗") | the category's measure word ("两只狗", "一台烤面包机") |

`configs/twins_v14.json` turns off the near-category pairs and the widened exclusion. The two label fixes always
apply, so the exact v1.4 questions are reproduced only by commit `23332a6`.

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

The mix itself is in the Hugging Face dataset `Vision-Decision/glance-train-data`, folder `v1.4/` (`train.jsonl`,
`stats_mp.json`, `gate_text.json`, `gate_image.json`); `v1.4_strict/` holds the variant without non-twinned LLM
negatives (k = 0 below). That dataset is **private**: reading it needs access to the `Vision-Decision` organisation,
or a public cleaned export. The data is not in this repository. The build statistics and gate summaries are in
`docs/results/twins_*.json`.

## The gate

`python -m glance_data.twins.gate` trains the probe on 90% of the strings of a mix and reports AUROC on the remaining
10%:

| Gate | Subset | Max | v1.3 mix | v1.4 mix |
|---|---|---:|---:|---:|
| S1 | rule-twinnable kinds | 0.65 | 0.954 | **0.500** |
| S2 | count questions | 0.55 | 0.948 | **0.500** |
| S3 | position questions | 0.55 | 0.753 | **0.500** |
| S4 | Chinese, rule-twinnable kinds | 0.55 | 0.959 | **0.500** |
| S5 (v1.5) | near-category negatives and their pairs | 0.55 | – | – |
| S1_full | every box question (reported) | – | 0.853 | 0.722 |
| S4_full | every Chinese question (reported) | – | 0.942 | 0.594 |

All four gates pass, and on every rule-twinnable subset the text-only probe is exactly at chance. That is by
construction: each string of these families occurs as often present as absent, and the split holds out whole
strings, so any text-only score gives both halves the same value. The gates therefore confirm that every rule-made
string got its twin and none was left unpaired. They do not show that the twins need more than category detection;
for that, the audit (`glance_data.audit`) has to sample `mp_*` questions.

**What remains is a difference in style between sources, not the negatives.** S1_full and S4_full stay above
chance. The obvious suspect is the LLM-written hard negatives (`hneg_attribute`, `hneg_relation`, ...; "a black cow"
when every cow is brown). They cannot be twinned by rule: a twin would need an image on which that exact sentence is
true, and annotations cannot decide that. But removing them does not lower the full-pool AUROC; it raises it. Keeping
only a fraction k of the non-twinned LLM negatives:

| k (kept) | 1 (v1.4) | 0.5 | 0.25 | 0.1 | 0 |
|---|---:|---:|---:|---:|---:|
| S1_full | 0.722 | 0.741 | 0.748 | 0.747 | 0.768 |
| S4_full | 0.594 | 0.594 | 0.596 | 0.598 | 0.593 |

The signal comes from whole families of questions whose wording style differs from the negatives. The largest block
is about 48k box questions from the natural-image, GUI and PixMo pools: 82% of them are positive, and the probe gives
them a mean presence of 0.82. Human-written RefCOCO positives (`ref_en`) get 0.77. Removing that signal would mean
deleting GUI and natural-image positives, which is not worth it. So the LLM negatives are kept (they teach attribute
and relation checking), the gate scores the rule-twinnable kinds, and the full pool is reported next to them.

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
minus dropped negatives, and appends the new records. All settings are in `configs/twins_v15.json` (the default);
`stats_twins.json` records `candidate_images`, which depends on `--coco`: the Makefile default `COCO_ANN` covers only
the 19,994 RefCOCO images, while v1.4 drew from the full `instances_train2014.json`.

## Limitations

- **The full pool is not at chance** (S1_full 0.722, S4_full 0.594). The remaining signal is style differences between
  sources, not the negatives: dropping all non-twinned LLM hard negatives raises S1_full to 0.768 (see above). Twins
  fix rule-made questions only.
- **COCO misses objects.** An absent twin can be wrong when the category is in the image but not annotated. This is
  the same risk as the existing `neg_near` negatives. A present twin is wrong in the opposite case: "two dogs" on
an image with two annotated dogs and a third one nobody annotated.
- **Category matching is lexical.** A Chinese sentence whose noun is not in the lexicon, and whose box matches no
  annotation, gets no twin (counted as `*_no_category` in the stats). Its category cannot be identified, so it is not
  safe to twin.
