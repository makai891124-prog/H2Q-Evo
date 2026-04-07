# Stage 7: Real-time Manifold Rendering

## Purpose

Render a formula-driven manifold as both frame images and a playable mp4 artifact.

## Input Contract

`stage6_formula_contract.json` from Stage 6.

Required fields:

- `derived.eta`
- `derived.trace_depth`
- `derived.half_order_delta`
- `render_plan.frames`
- `render_plan.fps`
- `render_plan.grid_resolution`

## Output Contract

Write `stage7_render_report.json`:

```json
{
  "stage": "7_realtime_manifold_rendering",
  "driver": "pyvista",
  "video_path": "spatial/paper_formula_manifold_demo.mp4",
  "frames_dir": "spatial/frames",
  "saved_png_frames": [
    "spatial/frames/frame_0000.png"
  ],
  "frame_metrics": {
    "count": 96,
    "curvature_proxy_mean": 0.123,
    "curvature_proxy_max": 0.987
  },
  "passed": true,
  "error": null
}
```

## Pass Criteria

- `video_path` exists and is non-empty
- At least one png frame exists in `frames_dir`
- `frame_metrics.count == render_plan.frames`
