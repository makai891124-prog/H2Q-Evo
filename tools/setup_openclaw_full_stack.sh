#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENCLAW_DIR="$ROOT/external/openclaw"
NODE22_BIN="/opt/homebrew/opt/node@22/bin"

export PATH="$NODE22_BIN:$PATH"

ensure_openclaw_source() {
  if [[ -d "$OPENCLAW_DIR" && -f "$OPENCLAW_DIR/package.json" ]]; then
    return 0
  fi

  echo "[setup] OpenClaw source not found, downloading tarball..."
  mkdir -p "$ROOT/external"
  rm -rf "$OPENCLAW_DIR"

  tmp_tgz="$ROOT/external/openclaw_main.tar.gz"
  curl -L "https://codeload.github.com/openclaw/openclaw/tar.gz/refs/heads/main" -o "$tmp_tgz"
  tar -xzf "$tmp_tgz" -C "$ROOT/external"
  rm -f "$tmp_tgz"
  mv "$ROOT/external/openclaw-main" "$OPENCLAW_DIR"
}

ensure_node_and_pnpm() {
  if ! command -v node >/dev/null 2>&1; then
    echo "ERROR: node is not installed. Install Node >= 22.12.0 first." >&2
    echo "       macOS example: brew install node@22" >&2
    exit 1
  fi

  node_version="$(node -v | sed 's/^v//')"
  node_major="$(echo "$node_version" | cut -d. -f1)"
  node_minor="$(echo "$node_version" | cut -d. -f2)"
  if [[ "$node_major" -lt 22 || ( "$node_major" -eq 22 && "$node_minor" -lt 12 ) ]]; then
    echo "ERROR: Node $node_version is too old. Need >= 22.12.0." >&2
    exit 1
  fi

  corepack enable >/dev/null 2>&1 || true
  corepack prepare pnpm@latest --activate >/dev/null 2>&1 || true
  if ! command -v pnpm >/dev/null 2>&1; then
    echo "ERROR: pnpm is not available after corepack activation." >&2
    exit 1
  fi
}

build_openclaw() {
  cd "$OPENCLAW_DIR"
  echo "[setup] Installing OpenClaw dependencies..."
  if ! pnpm install --frozen-lockfile; then
    pnpm install
  fi

  echo "[setup] Building OpenClaw UI and CLI artifacts..."
  pnpm ui:build
  pnpm build

  if [[ ! -f "$OPENCLAW_DIR/dist/entry.js" && ! -f "$OPENCLAW_DIR/dist/entry.mjs" ]]; then
    echo "ERROR: Build completed but dist/entry.(m)js was not found." >&2
    exit 1
  fi
}

ensure_openclaw_source
ensure_node_and_pnpm
build_openclaw

echo
echo "OpenClaw full stack is prepared at: $OPENCLAW_DIR"
echo "Next step: bash tools/run_openclaw_full_stack.sh"
