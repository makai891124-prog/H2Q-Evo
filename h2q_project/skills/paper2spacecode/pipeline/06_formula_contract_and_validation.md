# Stage 6: Formula Contract and Validation

## Purpose

Convert paper formula text into a machine-readable contract that can drive rendering and validation.

## Input Contract

```json
{
  "paper_formula_text": "string",
  "driver": "tribonacci_sl3z",
  "render_plan": {
    "frames": 96,
    "fps": 24,
    "grid_resolution": 160
  }
}
```

## Output Contract

Write `stage6_formula_contract.json`:

```json
{
  "stage": "6_formula_contract",
  "driver": "tribonacci_sl3z",
  "input": {
    "paper_formula_text": "string"
  },
  "derived": {
    "eta": 1.8392867552,
    "determinant": 1.0,
    "foliation_depth": 8.0,
    "trace_depth": 68.0,
    "half_order_delta": 12.345,
    "ring_size": 13.0
  },
  "validation": {
    "determinant_is_unitary": true,
    "foliation_depth_in_range": true,
    "trace_depth_positive": true,
    "contract_ready": true
  },
  "render_plan": {
    "frames": 96,
    "fps": 24,
    "grid_resolution": 160
  }
}
```

## Pass Criteria

- `validation.contract_ready == true`
- `render_plan.frames > 0`
- `render_plan.fps > 0`
- `render_plan.grid_resolution >= 32`
