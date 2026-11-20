# Pipeline entry points. Configure paths in .env (see .env.example).
OUT ?= out
PY ?= python
WORKERS ?= 32

.PHONY: help test lint analyze inventory generate assemble compare pipeline diag-set audit export-hf twins twin-gate

help:
	@echo "make test | lint"
	@echo "make analyze                       # diagnose current data + benchmarks -> $(OUT)/analysis.json"
	@echo "make pipeline SPLIT=train          # inventory -> generate -> assemble -> compare"
	@echo "make diag-set                      # 300 held-out val images -> $(OUT)/aug_refcoco_val.jsonl"
	@echo "make twins EVAL_IDS=... [MIX=...]  # minimal pairs (docs/TWINS.md) -> $(OUT)/twins/train.jsonl"
	@echo "make twin-gate [SPLIT_BY=image]    # text-only probe AUROC -> $(OUT)/twins/gate_$(SPLIT_BY).json"

test:
	$(PY) -m pytest -q

lint:
	ruff check glance_data tests

analyze:
	$(PY) -m glance_data.analyze --out $(OUT)/analysis.json

SPLIT ?= train
HF_SPLIT ?= train
LIMIT ?= 0
inventory:
	$(PY) -m glance_data.inventory --images $(SPLIT) --out $(OUT)/inv_$(SPLIT) --limit $(LIMIT)

generate:
	$(PY) -m glance_data.generate --inv $(OUT)/inv_$(SPLIT) --out $(OUT)/gen_$(SPLIT) --workers $(WORKERS)

assemble:
	$(PY) -m glance_data.assemble --inv $(OUT)/inv_$(SPLIT) --gen $(OUT)/gen_$(SPLIT)/gen.jsonl --out $(OUT)/aug_$(SPLIT).jsonl

compare:
	$(PY) -m glance_data.compare --orig $(JEV_PREP)/train.jsonl --aug $(OUT)/aug_$(SPLIT).jsonl --out $(OUT)/compare_$(SPLIT).json

pipeline: inventory generate assemble compare

diag-set:
	$(MAKE) inventory generate assemble SPLIT=refcoco_val LIMIT=300

export-hf:
	$(PY) -m glance_data.export_hf --src $(OUT)/aug_$(SPLIT).jsonl --split $(HF_SPLIT) --out $(OUT)/hf_repo/data

audit:
	$(PY) -m glance_data.audit sample --aug $(OUT)/aug_$(SPLIT).jsonl --inv $(OUT)/inv_$(SPLIT) --out $(OUT)/audit_$(SPLIT)

# minimal pairs (docs/TWINS.md). EVAL_IDS: json list / dict of lists of held-out image ids; never used as twin images
MIX ?= $(OUT)/aug_$(SPLIT).jsonl
COCO_ANN ?= $(JEV_DATA)/grefcoco/instances.json
TWIN_CONFIG ?= configs/twins_v15.json
SPLIT_BY ?= text
twins:
	@test -n "$(EVAL_IDS)" || (echo "set EVAL_IDS=<eval_image_ids.json>"; exit 1)
	$(PY) -m glance_data.twins.build --mix $(MIX) --coco $(COCO_ANN) --eval-ids $(EVAL_IDS) --config $(TWIN_CONFIG) --out $(OUT)/twins

twin-gate:
	$(PY) -m glance_data.twins.gate --mix $(OUT)/twins/train.jsonl --split $(SPLIT_BY) --config $(TWIN_CONFIG) --out $(OUT)/twins/gate_$(SPLIT_BY).json
