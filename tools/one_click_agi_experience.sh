#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
REPORTS="$ROOT/reports"
mkdir -p "$REPORTS"

OPENCLAW_URL="http://127.0.0.1:8011"
H2Q_URL="http://127.0.0.1:8000"

MODE="demo"
RUN_CROSS_PUBLIC=1
RUN_SELF_IMPROVEMENT=0
SKIP_SELF_IMPROVEMENT=0
SELF_IMPROVEMENT_SESSIONS=3
PREFER_DEEPSEEK_ASSIST=0
for arg in "$@"; do
  if [[ "$arg" == "--interactive" ]]; then
    MODE="interactive"
    RUN_SELF_IMPROVEMENT=1
  fi
  if [[ "$arg" == "--skip-cross-public" ]]; then
    RUN_CROSS_PUBLIC=0
  fi
  if [[ "$arg" == "--with-self-improvement" ]]; then
    RUN_SELF_IMPROVEMENT=1
  fi
  if [[ "$arg" == "--skip-self-improvement" ]]; then
    SKIP_SELF_IMPROVEMENT=1
    RUN_SELF_IMPROVEMENT=0
  fi
  if [[ "$arg" == "--prefer-deepseek-assist" ]]; then
    PREFER_DEEPSEEK_ASSIST=1
  fi
done

OPENCLAW_STARTED=0
H2Q_STARTED=0
OPENCLAW_PID=""
H2Q_PID=""

cleanup() {
  if [[ "$OPENCLAW_STARTED" -eq 1 && -n "$OPENCLAW_PID" ]]; then
    kill "$OPENCLAW_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$H2Q_STARTED" -eq 1 && -n "$H2Q_PID" ]]; then
    kill "$H2Q_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_openclaw() {
  for _ in $(seq 1 60); do
    if curl -fsS "$OPENCLAW_URL/openclaw/manifest" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_h2q() {
  for _ in $(seq 1 60); do
    if curl -fsS "$H2Q_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    # /health may be inconsistent in this repo, fallback to /chat probe.
    if curl -fsS "$H2Q_URL/chat" \
      -H 'Content-Type: application/json' \
      -d '{"prompt":"ping","max_tokens":8,"temperature":0.1,"use_das_arch":false}' >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

echo "[1] Checking Python environment"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: Python not found at $PY"
  exit 1
fi

echo "[2] Ensuring OpenClaw adapter service"
if curl -fsS "$OPENCLAW_URL/openclaw/manifest" >/dev/null 2>&1; then
  echo "OpenClaw adapter already ready at $OPENCLAW_URL"
else
  echo "Starting OpenClaw adapter on :8011"
  "$PY" "$ROOT/tools/openclaw_h2q_adapter.py" --serve --host 127.0.0.1 --port 8011 \
    >"$REPORTS/one_click_openclaw_adapter.log" 2>&1 &
  OPENCLAW_PID="$!"
  OPENCLAW_STARTED=1
  wait_openclaw || { echo "ERROR: OpenClaw adapter failed to start"; exit 1; }
fi

echo "[3] Ensuring core H2Q server"
if wait_h2q; then
  echo "Core H2Q server already ready at $H2Q_URL"
else
  echo "Starting core H2Q server on :8000"
  "$PY" -m uvicorn h2q_project.h2q_server:app --host 127.0.0.1 --port 8000 \
    >"$REPORTS/one_click_h2q_server.log" 2>&1 &
  H2Q_PID="$!"
  H2Q_STARTED=1
  wait_h2q || { echo "ERROR: core H2Q server failed to start"; exit 1; }
fi

TS="$(date +%s)"
STEP1_OUT="$REPORTS/one_click_step1_responses_${TS}.json"
STEP2_OUT="$REPORTS/one_click_step2_full_${TS}.json"

echo "[4] Step 1: /v1/responses protocol experience"
curl -fsS "$OPENCLAW_URL/v1/responses" \
  -H 'Content-Type: application/json' \
  -H 'x-openclaw-agent-id: main' \
  -d '{"model":"h2q-openclaw","input":[{"role":"user","content":[{"type":"input_text","text":"输出可执行的AGI三阶段演化计划"}]}]}' \
  >"$STEP1_OUT"
head -c 700 "$STEP1_OUT" || true
echo

echo "[5] Step 2: /openclaw/agent/run full chain"
curl -fsS "$OPENCLAW_URL/openclaw/agent/run" \
  -H 'Content-Type: application/json' \
  -d '{"task":"输出AGI系统实例化与公开验证方案","mode":"full","cycles":1}' \
  >"$STEP2_OUT"
head -c 1200 "$STEP2_OUT" || true
echo

if [[ "$MODE" == "interactive" ]]; then
  echo "Entering trusted interactive chat..."
  "$PY" "$ROOT/tools/trusted_local_agi_chat.py" --profile quick --skip-rsa --no-auto-start-server
else
  echo "Running trusted chat demo (auto prompt + exit)..."
  printf '请给我一个Python函数，实现two_sum，并附带pytest测试。\n/exit\n' | \
    "$PY" "$ROOT/tools/trusted_local_agi_chat.py" --profile quick --skip-rsa --no-auto-start-server
fi

if [[ "$RUN_SELF_IMPROVEMENT" -eq 1 && "$SKIP_SELF_IMPROVEMENT" -eq 0 ]]; then
  echo "[6] Running self-improvement closed loop"
  "$PY" "$ROOT/tools/run_self_improvement_closed_loop.py" --sessions "$SELF_IMPROVEMENT_SESSIONS"
fi

echo "[7] Generating one-click KPI dashboard"

# Always run the systemic joint assessment so each one-click run emits
# a cross-validated platform capability artifact.
if [[ "$RUN_CROSS_PUBLIC" -eq 1 ]]; then
  JOINT_BLUEPRINT_CYCLES=2
  JOINT_LONGRUN_CYCLES=2
else
  JOINT_BLUEPRINT_CYCLES=1
  JOINT_LONGRUN_CYCLES=1
fi

echo "[7] Running systemic joint capability assessment"
"$PY" "$ROOT/tools/run_systemic_platform_joint_capability_assessment.py" \
  --blueprint-cycles "$JOINT_BLUEPRINT_CYCLES" \
  --longrun-cycles "$JOINT_LONGRUN_CYCLES"

echo "[8] Generating one-click KPI dashboard"
KPI_ARGS=(
  "$ROOT/tools/generate_one_click_kpi_dashboard.py"
  --run-ts "$TS"
)
if [[ "$PREFER_DEEPSEEK_ASSIST" -eq 1 ]]; then
  KPI_ARGS+=(--prefer-deepseek-assist)
fi
"$PY" "${KPI_ARGS[@]}"

if [[ "$RUN_CROSS_PUBLIC" -eq 1 ]]; then
  echo "[9] Running auto blueprint + cross-public validation"
  "$PY" "$ROOT/tools/run_auto_blueprint_cross_public.py"

  echo "[10] Generating final demo scorecard"
  "$PY" "$ROOT/tools/generate_final_demo_scorecard.py"
else
  echo "[9] Skipped auto blueprint + cross-public validation (--skip-cross-public)"
  echo "[10] Skipped final demo scorecard (--skip-cross-public)"
fi

echo
echo "Done. Artifacts:"
echo "- $STEP1_OUT"
echo "- $STEP2_OUT"
echo "- reports/trusted_local_agi_chat_session_*.json (latest generated by step 3)"
echo "- reports/one_click_kpi_dashboard_latest.md"
echo "- reports/one_click_kpi_dashboard_latest.json"
echo "- reports/one_click_kpi_dashboard_latest.png"
echo "- reports/systemic_platform_joint_capability_latest.md"
echo "- reports/systemic_platform_joint_capability_latest.json"
if [[ "$RUN_CROSS_PUBLIC" -eq 1 ]]; then
  echo "- reports/auto_blueprint_cross_public_validation_latest.md"
  echo "- reports/final_demo_scorecard_latest.md"
fi
if [[ "$RUN_SELF_IMPROVEMENT" -eq 1 ]]; then
  echo "- reports/self_improvement_closed_loop_latest.md"
fi
echo
echo "Usage:"
echo "- default (full pipeline): $0"
echo "- interactive chat mode: $0 --interactive"
echo "- quick demo only: $0 --skip-cross-public"
echo "- full + self-improvement loop: $0 --with-self-improvement"
echo "- interactive without post self-improvement: $0 --interactive --skip-self-improvement"
echo "- prefer deepseek-assisted KPI source: $0 --prefer-deepseek-assist"
