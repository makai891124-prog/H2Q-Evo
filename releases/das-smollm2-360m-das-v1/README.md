---
language:
- en
license: apache-2.0
tags:
- das
- distillation
- smollm
- pytorch
library_name: pytorch
pipeline_tag: text-generation
---

# DAS Distilled Structure from SmolLM2-360M

This repository contains a distilled DAS (nonlinear algebraic geometric) structure package derived from `HuggingFaceTB/SmolLM2-360M`.

## What is included

- `das_token_structure_HuggingFaceTB__SmolLM2-360M_20260328.pt`: distilled DAS structure weights
- `das_token_structure_manifest_HuggingFaceTB__SmolLM2-360M_20260328.json`: structure manifest
- `loader.py`: minimal loader to inspect and run the distilled structure
- `metrics_single_run.json`: single-run metrics
- `metrics_multiseed.json`: multi-seed audit metrics
- `example_infer.py`: quick usage example

## Key performance summary

From `metrics_multiseed.json` (rank=32, seeds=11/22/33):

- mean cosine: 0.97005
- mean top5 overlap: 0.57321
- mean speedup ratio: 1.86966x
- mean compression ratio: 20.41705x

Threshold checks:

- consistency: PASS
- speedup: PASS
- compression: PASS

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python example_infer.py
```

## Notes

- This package is a DAS distilled structure package, not a full original Transformer checkpoint.
- Use with the provided loader and manifest.
- Claims are bounded to local tested scopes and should not be interpreted as hardware-level quantum-advantage replication.

## Source model and attribution

- Teacher model: `HuggingFaceTB/SmolLM2-360M`
- Please ensure compliance with upstream license and attribution requirements when redistributing derivatives.
