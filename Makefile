# Pipeline entry points. Configure paths in .env (see .env.example).
OUT ?= out
PY ?= python
WORKERS ?= 32

.PHONY: help test lint analyze inventory generate assemble compare pipeline diag-set audit export-hf

help:
	@echo "make test | lint"
	@echo "make analyze                       # diagnose current data + benchmarks -> $(OUT)/analysis.json"
	@echo "make pipeline SPLIT=train          # inventory -> generate -> assemble -> compare"
	@echo "make diag-set                      # 300 held-out val images -> $(OUT)/aug_refcoco_val.jsonl"

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
