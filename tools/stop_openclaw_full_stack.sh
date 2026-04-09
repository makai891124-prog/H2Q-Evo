#!/usr/bin/env bash
set -euo pipefail

OPENCLAW_GATEWAY_PORT="${OPENCLAW_GATEWAY_PORT:-18789}"
OPENCLAW_ADAPTER_PORT="${OPENCLAW_ADAPTER_PORT:-8011}"
H2Q_CORE_PORT="${H2Q_CORE_PORT:-8000}"

SKIP_GATEWAY=0
SKIP_ADAPTER=0
SKIP_H2Q_CORE=0
DRY_RUN=0
STOPPED_PIDS=""

for arg in "$@"; do
  if [[ "$arg" == "--skip-gateway" ]]; then
    SKIP_GATEWAY=1
  fi
  if [[ "$arg" == "--skip-adapter" ]]; then
    SKIP_ADAPTER=1
  fi
  if [[ "$arg" == "--skip-h2q-core" ]]; then
    SKIP_H2Q_CORE=1
  fi
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

stop_pids() {
  local name="$1"
  shift
  local pids=("$@")
  if [[ "${#pids[@]}" -eq 0 ]]; then
    return 0
  fi

  local uniq=()
  local pid
  for pid in "${pids[@]}"; do
    if [[ "$STOPPED_PIDS" == *" $pid "* ]]; then
      continue
    fi
    uniq+=("$pid")
  done

  if [[ "${#uniq[@]}" -eq 0 ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] Would stop $name PIDs: ${uniq[*]}"
    STOPPED_PIDS+=" ${uniq[*]} "
    return 0
  fi

  echo "[stop] Stopping $name PIDs: ${uniq[*]}"
  kill "${uniq[@]}" >/dev/null 2>&1 || true
  STOPPED_PIDS+=" ${uniq[*]} "
  sleep 1

  local survivors=()
  local pid
  for pid in "${uniq[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      survivors+=("$pid")
    fi
  done

  if [[ "${#survivors[@]}" -gt 0 ]]; then
    echo "[stop] Force-killing remaining $name PIDs: ${survivors[*]}"
    kill -9 "${survivors[@]}" >/dev/null 2>&1 || true
  fi
}

stop_by_port() {
  local name="$1"
  local port="$2"
  local pids
  pids="$(lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    echo "[stop] No $name listener on :$port"
    return 0
  fi
  # shellcheck disable=SC2206
  local pid_array=($pids)
  stop_pids "$name(:$port)" "${pid_array[@]}"
}

stop_by_pattern() {
  local name="$1"
  local pattern="$2"
  local pids
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  # shellcheck disable=SC2206
  local pid_array=($pids)
  stop_pids "$name(pattern)" "${pid_array[@]}"
}

if [[ "$SKIP_GATEWAY" -eq 0 ]]; then
  stop_by_port "openclaw-gateway" "$OPENCLAW_GATEWAY_PORT"
  stop_by_pattern "openclaw-gateway" "openclaw\\.mjs gateway run"
fi

if [[ "$SKIP_ADAPTER" -eq 0 ]]; then
  stop_by_port "h2q-openclaw-adapter" "$OPENCLAW_ADAPTER_PORT"
  stop_by_pattern "h2q-openclaw-adapter" "tools/openclaw_h2q_adapter.py --serve"
fi

if [[ "$SKIP_H2Q_CORE" -eq 0 ]]; then
  stop_by_port "h2q-core" "$H2Q_CORE_PORT"
  stop_by_pattern "h2q-core" "uvicorn h2q_project.h2q_server:app"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Completed preview only."
else
  echo "[stop] Full-stack stop completed."
fi
