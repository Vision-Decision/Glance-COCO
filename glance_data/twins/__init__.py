"""Minimal pairs ("twins") that remove text-only shortcuts from rule-made questions.

  build.py    pair every rule-made question with a same-string question of the other polarity on another image
  gate.py     text-only probe: AUROC of predicting `present` from the string alone, with a string-grouped split
  config.py   TwinConfig (defaults = v1.4); configs/twins_v14.json records them
  lexicon.py  category names (en/zh) used to parse and pair the rule templates
"""
