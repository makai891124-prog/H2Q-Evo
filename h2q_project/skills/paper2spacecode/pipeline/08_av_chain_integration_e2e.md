# Stage 8: AV Integration and End-to-End Validation

## Purpose

Integrate spatial manifold video with the existing audio/video cognition chain and produce one end-to-end report.

## Input Contract

```json
{
  "voice_chain_report_path": "reports/voice_video_dialogue_chain_report.json",
  "stage7_render_report_path": "spatial/stage7_render_report.json",
  "stage6_formula_contract_path": "contracts/stage6_formula_contract.json"
}
```

If local chain media is available, compose a side-by-side video.

## Output Contract

Write `stage8_e2e_report.json`:

```json
{
  "stage": "8_av_integration_e2e",
  "inputs": {
    "voice_chain_report_path": "string",
    "stage7_render_report_path": "string",
    "stage6_formula_contract_path": "string"
  },
  "outputs": {
    "combined_video_path": "integration/paper_formula_space_av_e2e.mp4",
    "formula_driver_text": "string"
  },
  "checks": {
    "voice_chain_passed": true,
    "stage7_passed": true,
    "combined_video_created": true
  },
  "passed": true,
  "error": null
}
```

## Pass Criteria

- Voice chain report exists and parses
- Stage 7 report exists and `passed == true`
- Combined video exists when chain video is present
- Final `passed == true`
