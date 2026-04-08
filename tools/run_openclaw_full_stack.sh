#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
OPENCLAW_DIR="$ROOT/external/openclaw"
REPORTS="$ROOT/reports"
NODE22_BIN="/opt/homebrew/opt/node@22/bin"

OPENCLAW_GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-18789}"
OPENCLAW_ADAPTER_PORT="${OPENCLAW_ADAPTER_PORT:-8011}"
H2Q_CORE_PORT="${H2Q_CORE_PORT:-8000}"
OPENCLAW_GATEWAY_TOKEN="${OPENCLAW_GATEWAY_TOKEN:-h2q-openclaw-local-token}"

SKIP_H2Q_CORE=0
for arg in "$@"; do
  if [[ "$arg" == "--skip-h2q-core" ]]; then
    SKIP_H2Q_CORE=1
  fi
done

export PATH="$NODE22_BIN:$PATH"
mkdir -p "$REPORTS"

wait_http_ok() {
  local url="$1"
  local timeout_sec="${2:-60}"
  local i
  for i in $(seq 1 "$timeout_sec"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_gateway_ok() {
  local timeout_sec="${1:-60}"
  local i
  for i in $(seq 1 "$timeout_sec"); do
    if (
      cd "$OPENCLAW_DIR" &&
        OPENCLAW_GATEWAY_TOKEN="$OPENCLAW_GATEWAY_TOKEN" \
          node openclaw.mjs gateway health \
          --url "ws://127.0.0.1:${OPENCLAW_GATEWAY_PORT}" \
          --token "$OPENCLAW_GATEWAY_TOKEN" \
          --timeout 2000 \
          --json
    ) >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

ensure_requirements() {
  if [[ ! -x "$PY" ]]; then
    echo "ERROR: Python venv not found at $PY" >&2
    exit 1
  fi
  if [[ ! -d "$OPENCLAW_DIR" ]]; then
    echo "ERROR: OpenClaw source not found at $OPENCLAW_DIR" >&2
    echo "Run: bash tools/setup_openclaw_full_stack.sh" >&2
    exit 1
  fi
  if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: node is not available." >&2
    exit 1
  fi
  if [[ ! -f "$OPENCLAW_DIR/dist/entry.js" && ! -f "$OPENCLAW_DIR/dist/entry.mjs" ]]; then
    echo "ERROR: OpenClaw dist artifacts missing. Run setup first." >&2
    exit 1
  fi
}

start_openclaw_gateway() {
  if lsof -nP -iTCP:"$OPENCLAW_GATEWAY_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "[gateway] Detected existing listener on :$OPENCLAW_GATEWAY_PORT"
    if wait_gateway_ok 5; then
      echo "[gateway] Existing OpenClaw gateway is healthy"
      return 0
    fi
    echo "[gateway] Existing listener is unhealthy for OpenClaw health probe"
    exit 1
  fi

  echo "[gateway] Starting OpenClaw gateway on :$OPENCLAW_GATEWAY_PORT"
  (
    cd "$OPENCLAW_DIR"
    OPENCLAW_GATEWAY_TOKEN="$OPENCLAW_GATEWAY_TOKEN" \
      OPENCLAW_SKIP_CHANNELS=1 \
      CLAWDBOT_SKIP_CHANNELS=1 \
      node openclaw.mjs gateway run \
      --allow-unconfigured \
      --bind loopback \
      --port "$OPENCLAW_GATEWAY_PORT" \
      --token "$OPENCLAW_GATEWAY_TOKEN" \
      --force \
      --verbose \
      >"$REPORTS/openclaw_gateway.log" 2>&1 &
  )

  if ! wait_gateway_ok 60; then
    echo "ERROR: OpenClaw gateway failed health check. See $REPORTS/openclaw_gateway.log" >&2
    exit 1
  fi
}

start_h2q_adapter() {
  local adapter_url="http://127.0.0.1:${OPENCLAW_ADAPTER_PORT}/openclaw/manifest"
  if wait_http_ok "$adapter_url" 2; then
    echo "[adapter] H2Q OpenClaw adapter already healthy on :$OPENCLAW_ADAPTER_PORT"
    return 0
  fi

  echo "[adapter] Starting H2Q OpenClaw adapter on :$OPENCLAW_ADAPTER_PORT"
  "$PY" "$ROOT/tools/openclaw_h2q_adapter.py" --serve --host 127.0.0.1 --port "$OPENCLAW_ADAPTER_PORT" \
    >"$REPORTS/openclaw_h2q_adapter.log" 2>&1 &

  if ! wait_http_ok "$adapter_url" 60; then
    echo "ERROR: H2Q adapter failed health check. See $REPORTS/openclaw_h2q_adapter.log" >&2
    exit 1
  fi
}

start_h2q_core() {
  local health_url="http://127.0.0.1:${H2Q_CORE_PORT}/health"
  if wait_http_ok "$health_url" 2; then
    echo "[h2q] Core H2Q server already healthy on :$H2Q_CORE_PORT"
    return 0
  fi

  echo "[h2q] Starting core H2Q server on :$H2Q_CORE_PORT"
  "$PY" -m uvicorn h2q_project.h2q_server:app --host 127.0.0.1 --port "$H2Q_CORE_PORT" \
    >"$REPORTS/h2q_core_server.log" 2>&1 &

  if ! wait_http_ok "$health_url" 60; then
    echo "ERROR: Core H2Q server failed health check. See $REPORTS/h2q_core_server.log" >&2
    exit 1
  fi
}

print_summary() {
  echo
  echo "OpenClaw + H2Q full stack is running."
  echo "- OpenClaw Gateway WS: ws://127.0.0.1:${OPENCLAW_GATEWAY_PORT}"
  echo "- OpenClaw Gateway token: ${OPENCLAW_GATEWAY_TOKEN}"
  echo "- H2Q OpenClaw Adapter: http://127.0.0.1:${OPENCLAW_ADAPTER_PORT}"
  if [[ "$SKIP_H2Q_CORE" -eq 0 ]]; then
    echo "- H2Q Core Server: http://127.0.0.1:${H2Q_CORE_PORT}"
  fi
  echo
  echo "Quick checks:"
  echo "- node external/openclaw/openclaw.mjs gateway health --url ws://127.0.0.1:${OPENCLAW_GATEWAY_PORT} --token ${OPENCLAW_GATEWAY_TOKEN} --json"
  echo "- curl -s http://127.0.0.1:${OPENCLAW_ADAPTER_PORT}/openclaw/manifest"
  echo "- curl -s http://127.0.0.1:${OPENCLAW_ADAPTER_PORT}/v1/responses -H 'Content-Type: application/json' -H 'x-openclaw-agent-id: main' -d '{\"model\":\"h2q-openclaw\",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"给出一个最小可执行自治计划\"}]}]}'"
}

ensure_requirements
start_openclaw_gateway
start_h2q_adapter
if [[ "$SKIP_H2Q_CORE" -eq 0 ]]; then
  start_h2q_core
fi
print_summary
