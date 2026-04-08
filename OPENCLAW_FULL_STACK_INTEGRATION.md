# OpenClaw Full Stack Integration (H2Q-Evo)

This document describes how to run full OpenClaw in this repository and wire it
with H2Q services for local agent workflows.

## What this enables

- Full OpenClaw gateway runtime from source (`external/openclaw`)
- H2Q OpenClaw-compatible adapter (`tools/openclaw_h2q_adapter.py`)
- Optional H2Q core inference service (`h2q_project/h2q_server.py`)

The adapter endpoint remains the canonical bridge for H2Q algorithm routes:

- `http://127.0.0.1:8011/openclaw/manifest`
- `http://127.0.0.1:8011/openclaw/agent/run`
- `http://127.0.0.1:8011/v1/responses`

## Prerequisites

- macOS with Node >= 22.12.0
- Python virtualenv at `.venv`

## Step 1: Prepare full OpenClaw runtime

```bash
bash tools/setup_openclaw_full_stack.sh
```

This step will:

- Ensure `external/openclaw` source exists (download tarball if needed)
- Ensure Node and pnpm are available
- Run `pnpm install`, `pnpm ui:build`, and `pnpm build`

## Step 2: Start integrated local stack

```bash
bash tools/run_openclaw_full_stack.sh
```

Default ports:

- OpenClaw Gateway: `18789`
- H2Q OpenClaw adapter: `8011`
- H2Q core server: `8000`

Optional flag:

- `--skip-h2q-core`: start gateway + adapter only

One-click integration path:

```bash
bash tools/one_click_agi_experience.sh --with-openclaw-full
```

One-click with automatic teardown at process exit:

```bash
bash tools/one_click_agi_experience.sh --with-openclaw-full --teardown-openclaw-full
```

One-click with forced teardown and stop-script logging:

```bash
bash tools/one_click_agi_experience.sh --with-openclaw-full --teardown-openclaw-full-force
```

Optional environment overrides:

- `OPENCLAW_GATEWAY_PORT`
- `OPENCLAW_ADAPTER_PORT`
- `H2Q_CORE_PORT`
- `OPENCLAW_GATEWAY_TOKEN`

## Step 3: Stop integrated local stack

```bash
bash tools/stop_openclaw_full_stack.sh
```

Optional stop flags:

- `--skip-gateway`
- `--skip-adapter`
- `--skip-h2q-core`
- `--dry-run`

## Verification commands

Gateway health:

```bash
node external/openclaw/openclaw.mjs gateway health \
  --url ws://127.0.0.1:18789 \
  --token h2q-openclaw-local-token \
  --json
```

Adapter manifest:

```bash
curl -s http://127.0.0.1:8011/openclaw/manifest
```

OpenResponses round-trip:

```bash
curl -s http://127.0.0.1:8011/v1/responses \
  -H 'Content-Type: application/json' \
  -H 'x-openclaw-agent-id: main' \
  -d '{"model":"h2q-openclaw","input":[{"role":"user","content":[{"type":"input_text","text":"给出一个最小可执行自治计划"}]}]}'
```

## Logs

Runtime logs are written to `reports/`:

- `reports/openclaw_gateway.log`
- `reports/openclaw_h2q_adapter.log`
- `reports/h2q_core_server.log`
- `reports/one_click_openclaw_full_stop.log` (when `--teardown-openclaw-full-force` is used)

## Notes

- OpenClaw CLI from source uses `node external/openclaw/openclaw.mjs ...`.
- This setup keeps full OpenClaw runtime and H2Q algorithm bridge separated so
  both can evolve independently.
