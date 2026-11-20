"""Settings for the minimal-pair builder and the text-only gate.

The defaults are the v1.4 settings; `configs/twins_v14.json` records the same values explicitly. A JSON file passed
with `--config` overrides any subset of them (unknown keys are an error, so a typo cannot silently fall back).
"""
import dataclasses
import json

# rule-answered positive kinds: the target set follows from annotations, so the same string is certainly absent on an
# image without the category (or, for count_pos, with a different count)
TWIN_KINDS = ["cat_only", "cat_all", "cf_anchor", "pos_pos_sole", "count_pos", "cf_func", "cf_generic", "cf_rel"]


@dataclasses.dataclass
class TwinConfig:
    seed: int = 14
    # --- builder
    twin_kinds: list = dataclasses.field(default_factory=lambda: list(TWIN_KINDS))
    zh_kinds: list = dataclasses.field(default_factory=lambda: ["ref_zh"])   # human RefCOCO sentences translated to zh
    piles: list = dataclasses.field(default_factory=lambda: ["book", "banana", "carrot", "broccoli", "apple", "orange",
                                                             "donut"])   # often only partly annotated: never count them
    count_min: int = 2
    count_max: int = 5
    min_area: float = 0.005          # an instance used as a positive covers at least this share of the image
    edge_px: float = 2.0             # a box within this many pixels of the border touches it ...
    edge_min_area: float = 0.02      # ... and is a cut-off sliver if it also covers less than this share
    side_frac: float = 1 / 3         # "on the left": the sole instance's centre is in the left third (right: right third)
    box_match_px: float = 2.0        # a source-image annotation within this distance of the target box is the target
    absent_tries: int = 50           # random draws when looking for an image without the named categories
    max_questions_per_record: int = 48   # new records are split into chunks of this many questions (0: one per image)
    # --- text-only gate
    probe_cap: int = 1_500_000       # subsample the questions to at most this many
    test_frac: float = 0.1
    count_kinds: list = dataclasses.field(default_factory=lambda: ["neg_count", "count_pos", "mp_count_pos"])
    position_kinds: list = dataclasses.field(default_factory=lambda: ["neg_position", "pos_pos_sole", "mp_pos_pos"])
    rule_kinds: list = dataclasses.field(default_factory=lambda: [
        "neg_count", "count_pos", "mp_count_pos", "neg_position", "pos_pos_sole", "mp_pos_pos", "ref_zh", "mp_zh_neg",
        "cat_only", "cat_all", "cf_anchor", "cf_func", "cf_generic", "cf_rel"])
    # gate name -> [subset, max AUROC]; S1_full / S4_full are reported, they are not expected to pass (see docs/TWINS.md)
    gates: dict = dataclasses.field(default_factory=lambda: {
        "S1": ["rule", 0.65], "S2": ["count", 0.55], "S3": ["position", 0.55], "S4": ["zh_rule", 0.55],
        "S1_full": ["all", 0.65], "S4_full": ["zh", 0.55]})


def load_config(path=None, **overrides):
    """TwinConfig from defaults, then the JSON file at `path`, then keyword overrides (e.g. from the command line)."""
    vals = {}
    if path:
        vals.update(json.load(open(path)))
    vals.update({k: v for k, v in overrides.items() if v is not None})
    names = {f.name for f in dataclasses.fields(TwinConfig)}
    unknown = sorted(set(vals) - names)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return TwinConfig(**vals)
