# AGENTS.md

Guide for coding agents (and people) working in this repository. Read it before changing code or running the pipeline.

## Scope: data only

This repository builds and checks **data** for a Jev-style grounding task. Each example is one image plus many text
requests, and each request is answered with a box, several boxes, or "not present".

In scope:
1. **Diagnose** the existing data (RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO as prepared for training) and the
   benchmark sets.
2. **Augment** it. Exact targets come from COCO annotations. A vision LLM (DeepSeek) only writes the wording.
3. **Validate the data itself**: rule filters, and a stratified human audit of label precision.
4. **Produce data products**: the augmented training records and a diagnostic set on held-out images.

Out of scope: model code, training, checkpoints, and running models to evaluate data. Those live in the model
repository, which consumes the files this repo produces. Do not add model or training dependencies here.

## Layout

```
glance_data/
  common.py      config (env vars), loaders for RefCOCO*/gRefCOCO/COCO, text heuristics
  analyze.py     stage 0  diagnose current data + benchmarks              -> out/analysis.json
  inventory.py   stage 1  per-image object inventory + set-of-mark images  -> out/inv_<split>/
  generate.py    stage 2  one DeepSeek vision call per image              -> out/gen_<split>/gen.jsonl
  assemble.py    stage 3  rule filters + rule-made queries + merge        -> out/aug_<split>.jsonl
  compare.py     before/after distribution of the training data           -> out/compare_<split>.json
  audit.py       stratified human audit: sample + render, then score
  export_hf.py   records -> Parquet shards for the Hugging Face dataset (no images)
  twins/         minimal pairs: same string, other answer, other image   -> out/twins/ (docs/TWINS.md)
    build.py     count / position / Chinese pairs and missing-category twins
    gate.py      text-only probe gate (AUROC, string-grouped split)
configs/twins_v14.json  the v1.4 minimal-pair and gate settings
hf/DATASET_CARD.md  the dataset card published as README.md on the Hub
tests/           unit tests for the rule-based logic (no data, no network)
assets/          logo (light/dark) and README artwork
docs/RESULTS.md  findings and audit numbers; docs/results/*.json raw summaries
```

## Setup

```bash
uv venv && uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
cp .env.example .env && set -a && source .env && set +a
pre-commit install                         # ruff + hygiene hooks on every commit
make test
```

On the server the sources live under `~/experiments/jevbox_20261117/`, in `data/` and `v2/data/prep/`. The defaults in
`glance_data/common.py` point there.

## Running

```bash
make analyze                          # stage 0
make pipeline SPLIT=train             # stages 1-3 + compare on all training images
make diag-set                         # 300 held-out refcoco_val images -> out/aug_refcoco_val.jsonl
python -m glance_data.audit sample --aug out/aug_train.jsonl --inv out/inv_train --out out/audit
python -m glance_data.audit score --judgements out/audit/judgements.json
make twins MIX=out/aug_train.jsonl EVAL_IDS=<ids.json>   # minimal pairs -> out/twins/train.jsonl
make twin-gate                        # text-only probe; needs pip install -e ".[probe]"
```

All stages are resumable. `generate` skips image ids that are already in `gen.jsonl`, so you can simply re-run it
after an interruption. Run long jobs in a dedicated tmux session named `glance-*`.

## Publishing (Hugging Face)

The dataset is `Vision-Decision/Glance-Data` on the Hub. It is **private**, licensed CC BY-NC-SA 4.0 (non-commercial
research only; gRefCOCO's ShareAlike terms carry over, see `LICENSE-DATA.md`), and contains no images. To publish a new version:

```bash
python -m glance_data.export_hf --src out/aug_train_all.jsonl --split train --out hf_repo/data
python -m glance_data.export_hf --src out/aug_refcoco_val.jsonl --split diagnostic --out hf_repo/data
cp hf/DATASET_CARD.md hf_repo/README.md        # update the numbers in it first
HF_TOKEN=$(cat <token file>) python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='Vision-Decision/Glance-Data', repo_type='dataset', folder_path='hf_repo')"
```

Use a token file outside the repo, as with the DeepSeek key, and pass it only through `HF_TOKEN`. Never run
`hf auth login` on the shared server: it would replace the login other projects use. Ask before making the dataset
public.

## Record schema (the contract with consumers)

The per-image layout is the same as the prepared `train.jsonl`: `image_id, file, W, H, questions`. Each question has:

| field | meaning |
|---|---|
| `text` | the query |
| `present` | bool |
| `box` | xyxy pixels of the single target, or `null` |
| `boxes` | all targets (xyxy), `null` when absent |
| `n_targets` | 0, 1 or more |
| `kind` | `short medium long casual group function hneg_attribute hneg_relation cat_only cat_all neg_count neg_position neg_near gref_multi`; minimal pairs add `mp_count_pos mp_pos_pos mp_zh_neg mp_twin_neg` |
| `lang` | `en` or `zh` |
| `src` | `aug_llm`, `aug_function`, `aug_code`, `aug_gref` or `aug_twin` |
| `twin_of` | only on `mp_twin_neg` and `mp_zh_neg`: the kind of the present question this one twins |

Changing this schema breaks consumers. If you add a field, keep the existing ones and document the new one here.
Multi-target questions (`n_targets > 1`) need a consumer that supports several boxes.

## Design rules (do not break these)

1. **Truth from annotations, wording from the LLM.** Boxes, counts and "is this category in the image" are computed
   from COCO annotations. The LLM never decides a target set, except by naming marked objects or COCO categories.
   Every new query kind must document where its ground truth comes from.
2. **No extra models.** Do not deploy or download models to check or filter data. A local VLM verifier was tried and
   rejected on 2026-11-18. Quality control is rule filters plus the stratified human audit.
3. **Rule filters live in `assemble.py`.** Currently:
   - drop text that leaks the set-of-mark drawing (box, mark, 框, 编号, ...);
   - drop hard negatives whose position words differ from the source medium description (the LLM sometimes changes the
     position instead of the attribute);
   - no category-only query when the largest instance covers less than 0.5% of the image;
   - drop functions that touch crowd-annotated categories.
4. **Rule-made questions come in pairs.** `twins/build.py` gives every rule-made negative a present question with the
   same string on another image (and drops the negative when no such image exists), and every rule-answered positive
   an absent one. Ground truth: COCO counts, the sole instance's side, and category absence. Check a new mix with
   `make twin-gate`; S1-S4 must stay within their thresholds (docs/TWINS.md).
5. **Held-out images stay held out.** Training augmentation uses only images in `JEV_PREP/train.jsonl`, which has every
   benchmark image removed. The diagnostic set uses `refcoco_val` images. Never mix the two.
6. **DeepSeek usage.** Use `deepseek-flash` with thinking disabled (`thinking: {type: disabled}`): about 0.005 CNY per
   image (the full 24,406-image run cost ~122 CNY and took ~70 min with 32 workers). Thinking mode costs roughly 30x more and times out on long answers. Check
   the balance (`GET /user/balance`) before runs of more than a few thousand images.
7. **Secrets.** The API key lives only in a file outside the repo (`DEEPSEEK_API_KEY_FILE`, chmod 600). Never print it,
   pass it on a command line or commit it. CI fails on anything that looks like a key. Never commit server addresses
   or SSH keys.

## Validation protocol

- After any change to the prompt or a filter, regenerate a small sample (about 150 training images), run
  `audit sample` with 8 items per kind, judge the sheets by eye, run `audit score`, and record the per-kind numbers in
  `docs/RESULTS.md`.
- A kind whose precision drops below 6/8 is either fixed or left out with `assemble --drop-kinds`.
- `compare.py` must show the intended shift: more category-only, multi-target, same-category negatives and zh items.

## Shared server etiquette

The server is a shared 8×H200 machine with live services from other projects.

- This repo needs no GPU. Run with `nice -n 10` in your own tmux session.
- Never kill, restart or reconfigure processes, services or virtualenvs you did not start.
- Write only under `~/experiments/jevbox_20261117/`. The disk is above 90% full, so delete your own large
  intermediates when you are done.

## Code conventions

- Python ≥3.10. Dependencies are the standard library plus numpy, pandas, pyarrow and pillow. Nothing else. The one
  exception is scikit-learn for the text-only gate, as the optional `probe` extra; nothing else may import it.
- Every stage is a module with a CLI (`python -m glance_data.<stage>`), resumable, and deterministic given `--seed`.
- Keep rule logic as pure functions (`assemble.code_items`, `assemble.spatial_sig`, `generate.collect_queries`) and
  cover them in `tests/`. Tests must not need data, GPUs or the network.
- `ruff check glance_data tests` must pass. Keep code lines to about 120 characters; long prompt text is exempt.

## Git

Work on a branch and open a PR to `main`. Never commit outputs (`out/`, `*.jsonl`, images); small result summaries go
in `docs/results/`. Commit messages are imperative and short, with a body that explains why.

## Known limitations

- Short descriptions are sometimes not unique (6/8 in the audit), e.g. "green tie" when there are two green ties.
- LLM-written hard negatives (`hneg_*`) cannot be twinned by rule, so the text-only AUROC over all questions stays above
  chance (0.722 on the v1.4 mix) even though every rule-made family is at 0.5.
- COCO misses some objects, so a rule-made negative such as "a toaster" can be wrong when the object is present but
  unannotated (near-category negatives: 7/8 in the audit).
- Licensing: code is Apache-2.0; all data is CC BY-NC-SA 4.0 for non-commercial research only (gRefCOCO is
  CC BY-NC-SA 4.0 and its ShareAlike clause carries over). Do not relicense data or add sources with stricter terms
  without updating `LICENSE-DATA.md`.
