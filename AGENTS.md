# AGENTS.md

Guide for coding agents (and people) working in this repository. Read it before changing code or running the pipeline.

## What this repo is

Data for a Jev-style grounding model: one image plus many text requests, answered in parallel with a box, several
boxes, or "not present". This repo **only handles data**. Model training lives in the vlm-jev repository.

The repo does three things:
1. **Diagnose** the existing training data (RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO, as prepared by vlm-jev) and
   the benchmarks.
2. **Augment** it. Exact targets come from COCO annotations. A vision LLM (DeepSeek) only writes the wording.
3. **Validate** the new data in two ways: a stratified human audit of label precision, and a diagnostic set on held-out
   images, evaluated with a vlm-jev checkpoint.

## Layout

```
glance_data/
  common.py        config (env vars), loaders for RefCOCO*/gRefCOCO/COCO, text heuristics
  analyze.py       stage 0  diagnose current data + benchmarks            -> out/analysis.json
  inventory.py     stage 1  per-image object inventory + set-of-mark images -> out/inv_<split>/
  generate.py      stage 2  one DeepSeek vision call per image            -> out/gen_<split>/gen.jsonl
  assemble.py      stage 3  rule filters + rule-made queries + merge      -> out/aug_<split>.jsonl
  compare.py       before/after distribution of the training data
  eval/            needs a vlm-jev checkout (VLM_JEV_ROOT)
    model_rows.py    per-question predictions on benchmark sets
    audit_failures.py  render random failures (ground truth vs prediction)
    diagnose.py      per-kind metrics of a checkpoint on an augmented record file
tests/             unit tests for the rule-based logic (no data, no network)
docs/RESULTS.md    findings, audit and diagnostic numbers; docs/results/*.json raw results
```

## Setup

```bash
uv venv && uv pip install -e ".[dev]"     # or: pip install -e ".[dev]"
cp .env.example .env && set -a && source .env && set +a
make test
```

On the server the data lives under `~/experiments/jevbox_20261117/` (`data/`, `v2/data/prep/`). Defaults in
`glance_data/common.py` point there, so the env vars matter mainly when you run somewhere else.

## Running

```bash
make analyze                          # stage 0
make pipeline SPLIT=train             # stages 1-3 + compare on all training images
make diag-set                         # 300 held-out refcoco_val images -> out/aug_val.jsonl
VLM_JEV_ROOT=... python -m glance_data.eval.diagnose --ckpt CKPT --records out/aug_val.jsonl --out out/diag.json
```

All stages are resumable. `generate` skips image ids already in `gen.jsonl`, so re-run it after an interruption.
Long runs go in a dedicated tmux session named `glance-*`.

## Record schema

The output has the same per-image layout as vlm-jev `train.jsonl` (`image_id, file, W, H, questions`). Each question:

| field | meaning |
|---|---|
| `text` | the query |
| `present` | bool |
| `box` | xyxy pixels of the single target, or `null` |
| `boxes` | all targets (xyxy). `null` when absent |
| `n_targets` | 0, 1 or more |
| `kind` | `short medium long casual group function hneg_attribute hneg_relation cat_only cat_all neg_count neg_position neg_near gref_multi` |
| `lang` | `en` or `zh` |
| `src` | `aug_llm`, `aug_function`, `aug_code` or `aug_gref` |

Multi-target questions need a multi-box head in the model. A single-box head can only use one target.

## Design rules (do not break these)

1. **Truth from annotations, wording from the LLM.** Boxes, counts and "is this category in the image" are computed
   from COCO annotations. The LLM never decides a target set except by naming marked objects or COCO categories.
   A new query kind must state where its ground truth comes from.
2. **No extra models for verification.** On 2026-11-18 we decided not to deploy local VLM verifiers (vLLM, Qwen
   9B/27B). Quality control is rule filters plus a stratified human audit. Do not reintroduce a verifier model without
   asking.
3. **Rule filters live in `assemble.py`.** Current filters:
   - drop text that leaks the set-of-mark drawing (box, mark, 框, 编号, ...);
   - drop hard negatives whose position words differ from the source medium description (the LLM sometimes "edits"
     position instead of attribute);
   - no category-only query when the largest instance is under 0.5% of the image;
   - functions touching crowd-annotated categories are dropped.
4. **Held-out images stay held out.** Training augmentation uses only images in `JEV_PREP/train.jsonl`, which has
   every benchmark image removed. The diagnostic set uses `refcoco_val` images. Never mix them.
5. **DeepSeek usage.** Use `deepseek-flash` with thinking disabled (`thinking: {type: disabled}`, ~0.003 CNY per
   image, ~5 images/s with 32 workers). Thinking mode costs ~30x more and times out on long answers. Check the balance
   (`GET /user/balance`) before runs of more than a few thousand images.
6. **Secrets.** The API key lives only in a file outside the repo (`DEEPSEEK_API_KEY_FILE`, chmod 600). Never print
   it, pass it on a command line, or commit it. Run `git grep -nE "sk-[A-Za-z0-9]{20,}"` before every push. Never
   commit server addresses or SSH keys.

## Validation protocol

- **Label precision:** sample 8 queries per kind, render them (positives with boxes, negatives without), and judge by
  eye whether text and answer match. Record per-kind counts in `docs/RESULTS.md`. The first audit was 101/112.
- **Diagnostic set:** `diagnose.py` reports, per kind and language, `acc50` (single target), `hit_any` (multi),
  `present_rate` and `reject_rate` (negatives). Compare against the same images with the original human text
  (`model_rows.py` on `refcoco_val`).
- After changing a filter or the prompt, regenerate the 300-image diagnostic set and re-audit the affected kinds.

## Shared GPU server etiquette

The server is a shared 8×H200 machine with live services from other projects on every GPU.

- Check `nvidia-smi` and use an idle GPU via `CUDA_VISIBLE_DEVICES`. Run with `nice -n 10` in a tmux session.
- Never kill, restart or reconfigure processes, services or virtualenvs you did not start.
- Write only under `~/experiments/jevbox_20261117/`. Disk is above 90% full, so delete your own large intermediates
  when done.

## Code conventions

- Python ≥3.10. Runtime deps are the standard library plus numpy, pandas, pyarrow and pillow (torch only in `eval/`).
- Every stage is a module with a CLI (`python -m glance_data.<stage>`), resumable and deterministic given `--seed`.
- Keep rule logic as pure functions (see `assemble.code_items`, `assemble.spatial_sig`, `generate.collect_queries`)
  and cover them in `tests/`. Tests must not need data, GPUs or the network.
- `ruff check glance_data tests` must pass; keep code lines to about 120 characters (long prompt text is exempt).

## Git

Work on a branch and open a PR to `main`. Do not commit outputs (`out/`, `*.jsonl`, images). Put small result
summaries in `docs/results/` instead. Commit messages are imperative and short, with a body explaining why.

## Known limitations

- Short descriptions are occasionally not unique (6/8 in the audit), e.g. "green tie" when there are two.
- COCO misses some objects, so a rule-made negative such as "a toaster" can be wrong when the object is present
  but unannotated. The rate is low (7/8 for near-category).
- gRefCOCO is licensed for non-commercial use only. Check this before any commercial use of `gref_multi` items.
