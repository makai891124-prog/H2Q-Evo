---
name: paper2spacecode
description: Extends paper2code with Stage 6/7/8 for formula contracts, real-time manifold visualization, and AV-integrated end-to-end validation.
---

# paper2spacecode - Minimal Orchestration

This skill starts after paper2code Stage 1-5 and adds three execution stages:

1. Stage 6 - Formula Contract and Verification
2. Stage 7 - Real-time Manifold Rendering (images + mp4)
3. Stage 8 - AV Chain Integration and End-to-End Validation

## Inputs

- `paper_slug` from paper2code output
- `paper_formula_text` (equations and key formula narrative)
- Optional `voice_chain_report` from local AV cognition pipeline

## Outputs

- `{paper_slug}/contracts/stage6_formula_contract.json`
- `{paper_slug}/spatial/stage7_render_report.json`
- `{paper_slug}/spatial/frames/*.png`
- `{paper_slug}/spatial/paper_formula_manifold_demo.mp4`
- `{paper_slug}/integration/stage8_e2e_report.json`
- `{paper_slug}/integration/paper_formula_space_av_e2e.mp4` (if AV chain video exists)

## Stage Dispatch

Read and execute in order:

- `pipeline/06_formula_contract_and_validation.md`
- `pipeline/07_realtime_manifold_rendering.md`
- `pipeline/08_av_chain_integration_e2e.md`

Do not skip stage order. Stage 7 depends on Stage 6 contract, Stage 8 depends on Stage 7 artifacts.
