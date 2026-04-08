# OpenClaw x H2Q Adapter Guide

This repository now includes an OpenClaw-compatible adapter aligned to the real
`openclaw/openclaw` integration style:

- Script: `tools/openclaw_h2q_adapter.py`
- HTTP API: `/openclaw/agent/run`, `/openclaw/manifest`
- OpenAI-compatible endpoint: `/v1/chat/completions`
- OpenResponses endpoint: `/v1/responses`

## Why this adapter

It provides OpenClaw-like usability while keeping H2Q core algorithm strengths:

- Local autonomous agent execution via `LocalExecutor` (precision-gated reasoning)
- Dynamic blueprint self-evolution via `tools/dynamic_blueprint_bootstrap.py`
- Public release validation via `tools/release_gate.py`

## Quick CLI usage

Run full integrated flow (agent + evolve + validate):

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/openclaw_h2q_adapter.py \
  --mode full \
  --task "请输出一个本地AGI系统实例化与公开验证方案" \
  --cycles 2 \
  --json
```

Run only agent task:

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/openclaw_h2q_adapter.py \
  --mode agent \
  --task "给出一个三阶段自治演化计划"
```

Run only evolution:

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/openclaw_h2q_adapter.py --mode evolve --cycles 4
```

Run only gate validation:

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/openclaw_h2q_adapter.py --mode validate
```

## HTTP service mode

Start server:

```bash
/Users/imymm/H2Q-Evo/.venv/bin/python tools/openclaw_h2q_adapter.py --serve --port 8011
```

Get capability manifest:

```bash
curl -s http://127.0.0.1:8011/openclaw/manifest
```

Run task in OpenClaw-style endpoint:

```bash
curl -s http://127.0.0.1:8011/openclaw/agent/run \
  -H 'Content-Type: application/json' \
  -d '{"task":"输出AGI自举路线", "mode":"full", "cycles":2}'
```

OpenAI-compatible request (OpenClaw supports this compatibility path):

```bash
curl -s http://127.0.0.1:8011/v1/chat/completions \
  -H 'Authorization: Bearer dev-token' \
  -H 'x-openclaw-agent-id: main' \
  -H 'Content-Type: application/json' \
  -d '{"model":"h2q-openclaw", "messages":[{"role":"user","content":"请给出自治演化计划"}]}'
```

OpenResponses request (preferred modern OpenClaw-compatible pattern):

```bash
curl -s http://127.0.0.1:8011/v1/responses \
  -H 'Authorization: Bearer dev-token' \
  -H 'x-openclaw-agent-id: main' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"h2q-openclaw",
    "input":[
      {
        "role":"user",
        "content":[{"type":"input_text","text":"输出可执行的AGI三阶段演化计划"}]
      }
    ]
  }'
```

## Mapping to H2Q core algorithms

- `agent`: `h2q_project.local_executor.LocalExecutor`
- `evolve`: `tools/dynamic_blueprint_bootstrap.py` with strong release-gate retries
- `validate`: `tools/release_gate.py` with public thresholds
- `full`: run all three in sequence and return merged artifacts

## Notes

- The adapter is designed to be integration-friendly, so OpenClaw-side calls only need one endpoint and one JSON payload.
- Artifacts are generated in `reports/` and returned in `artifacts` list.
- Routing header compatibility: `x-openclaw-agent-id` is accepted and echoed in response metadata.
