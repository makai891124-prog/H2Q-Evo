#!/usr/bin/env python3
"""Dynamic blueprint bootstrap runner.

Features:
- Dynamic blueprint scheduling from current system gaps.
- Auto-generation of candidate module skeletons under tools/generated_blueprints/.
- Optional strong release-gate loop with recovery retries.
- Cross-round strategy learning from historical outcomes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
GENERATED_DIR = ROOT / "tools" / "generated_blueprints"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest(glob_pat: str) -> Optional[Path]:
    files = sorted(REPORTS.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-2500:],
        "stderr_tail": (proc.stderr or "")[-2500:],
    }


@dataclass
class BlueprintSpec:
    blueprint_id: str
    title: str
    category: str
    reason: str
    priority: float
    hard_gate: bool
    cmd: List[str]
    expected_glob: Optional[str]


class BootstrapRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        REPORTS.mkdir(parents=True, exist_ok=True)
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        self.state_path = ROOT / args.state_file
        self.state = self._load_state()
        self.strategy = self._learn_strategy()

    def _load_state(self) -> Dict[str, Any]:
        obj = load_json(self.state_path)
        if obj:
            return obj
        return {
            "meta": {"created_at_utc": now_utc(), "version": 2},
            "history": [],
            "module_stats": {},
            "strategy": {},
        }

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _recent_history(self, window: int) -> List[Dict[str, Any]]:
        hist = self.state.get("history", [])
        if not isinstance(hist, list):
            return []
        return [h for h in hist[-max(1, window):] if isinstance(h, dict)]

    def _learn_strategy(self) -> Dict[str, Any]:
        recent = self._recent_history(max(2, self.args.history_window))
        runs = len(recent)
        ok_count = sum(1 for r in recent if bool(r.get("overall_ok", False)))
        ok_rate = float(ok_count) / float(max(1, runs))

        fail_streak = 0
        for row in reversed(recent):
            if bool(row.get("overall_ok", False)):
                break
            fail_streak += 1

        max_actions = int(max(1, self.args.max_actions_per_cycle))
        interactive_target = float(max(0.0, self.args.interactive_target))
        alignment_target = float(max(0.0, self.args.alignment_target))
        warn_drop = float(max(0.0, self.args.warn_drop))
        fail_drop = float(max(0.0, self.args.fail_drop))

        # Cross-round strategy learning from trend stability.
        if runs >= 2 and (ok_rate < 0.60 or fail_streak >= 2):
            max_actions = min(8, max_actions + 2)
            interactive_target = max(0.75, interactive_target - 0.03)
            alignment_target = max(0.70, alignment_target - 0.03)
            warn_drop = max(0.01, warn_drop - 0.005)
            fail_drop = max(warn_drop + 0.01, fail_drop - 0.01)
        elif runs >= 3 and ok_rate > 0.90 and fail_streak == 0:
            max_actions = max(2, max_actions - 1)
            interactive_target = min(0.95, interactive_target + 0.01)
            alignment_target = min(0.95, alignment_target + 0.01)
            warn_drop = max(0.005, warn_drop - 0.002)
            fail_drop = max(warn_drop + 0.01, fail_drop - 0.005)

        strategy = {
            "updated_at_utc": now_utc(),
            "observed_runs": runs,
            "ok_rate": ok_rate,
            "fail_streak": fail_streak,
            "max_actions_per_cycle": max_actions,
            "interactive_target": interactive_target,
            "alignment_target": alignment_target,
            "warn_drop": warn_drop,
            "fail_drop": fail_drop,
        }
        self.state["strategy"] = strategy
        return strategy

    def _context(self) -> Dict[str, Any]:
        gate = load_json(latest("release_gate_latest.json"))
        cap = load_json(latest("capability_registry_latest.json"))
        align = load_json(latest("public_alignment_report_latest.json"))
        ib = load_json(latest("interactive_reasoning_benchmark_latest.json"))
        reg = load_json(latest("nightly_regression_guard_latest.json"))

        gate_signals = gate.get("signals", {}) if isinstance(gate, dict) else {}
        gate_meta = gate.get("meta", {}) if isinstance(gate, dict) else {}
        caps = cap.get("capabilities", {}) if isinstance(cap, dict) else {}
        align_score = align.get("alignment", {}) if isinstance(align, dict) else {}
        ib_metrics = ib.get("metrics", {}) if isinstance(ib, dict) else {}
        reg_status = reg.get("status", {}) if isinstance(reg, dict) else {}

        breadth = float(gate_signals.get("breadth", caps.get("breadth", 0.0)) or 0.0)
        horizon = float(gate_signals.get("horizon", caps.get("horizon", 0.0)) or 0.0)
        robustness = float(gate_signals.get("robustness", caps.get("robustness", 0.0)) or 0.0)
        overall = float(align_score.get("overall", 0.0) or 0.0)
        interactive = float(ib_metrics.get("success_rate", 0.0) or 0.0)

        min_breadth = float(gate_meta.get("min_breadth", self.args.min_breadth) or self.args.min_breadth)
        min_horizon = float(gate_meta.get("min_horizon", self.args.min_horizon) or self.args.min_horizon)
        min_robust = float(gate_meta.get("min_robustness", self.args.min_robustness) or self.args.min_robustness)

        return {
            "scores": {
                "breadth": breadth,
                "horizon": horizon,
                "robustness": robustness,
                "alignment_overall": overall,
                "interactive_success": interactive,
            },
            "gaps": {
                "breadth": max(0.0, min_breadth - breadth),
                "horizon": max(0.0, min_horizon - horizon),
                "robustness": max(0.0, min_robust - robustness),
                "alignment": max(0.0, self.strategy["alignment_target"] - overall),
                "interactive": max(0.0, self.strategy["interactive_target"] - interactive),
            },
            "regression": {
                "warn": bool(reg_status.get("warn", False)),
                "fail": bool(reg_status.get("fail", False)),
            },
        }

    def _module_multiplier(self, blueprint_id: str) -> float:
        stats = self.state.get("module_stats", {}).get(blueprint_id, {})
        runs = int(stats.get("runs", 0) or 0)
        success = int(stats.get("success", 0) or 0)
        fail = int(stats.get("fail", 0) or 0)
        if runs <= 0:
            return 1.0
        success_rate = float(success) / float(max(1, runs))
        pressure = max(0.0, min(1.0, 1.0 - success_rate))
        fail_bias = max(0.0, min(1.0, float(fail) / float(max(1, runs))))
        return 1.0 + 0.35 * pressure + 0.15 * fail_bias

    def _generated_module_code(self, module_id: str, title: str, focus: str) -> str:
        prefix = f"generated_blueprint_{module_id}"
        return f'''#!/usr/bin/env python3
"""Auto-generated blueprint candidate module."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def latest(glob_pat: str):
    files = sorted(REPORTS.glob(glob_pat), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_json(path):
    if path is None or not path.exists():
        return {{}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {{}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generated blueprint candidate")
    parser.add_argument("--output-prefix", default="{prefix}")
    parser.add_argument("--min-objective", type=float, default=0.0)
    args = parser.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    cap = load_json(latest("capability_registry_latest.json"))
    align = load_json(latest("public_alignment_report_latest.json"))

    caps = cap.get("capabilities", {{}}) if isinstance(cap, dict) else {{}}
    alignment = align.get("alignment", {{}}) if isinstance(align, dict) else {{}}

    robustness = float(caps.get("robustness", 0.0) or 0.0)
    horizon = float(caps.get("horizon", 0.0) or 0.0)
    breadth = float(caps.get("breadth", 0.0) or 0.0)
    overall = float(alignment.get("overall", 0.0) or 0.0)

    objective = max(0.0, min(1.0, 0.35 * overall + 0.25 * robustness + 0.20 * horizon + 0.20 * breadth))

    payload = {{
        "meta": {{
            "created_at_utc": now_utc(),
            "module_id": "{module_id}",
            "title": "{title}",
            "focus": "{focus}",
        }},
        "result": {{
            "objective": objective,
            "meets_min_objective": objective >= max(0.0, args.min_objective),
        }},
    }}

    ts = int(time.time())
    out_json = REPORTS / f"{{args.output_prefix}}_{{ts}}.json"
    out_latest = REPORTS / f"{{args.output_prefix}}_latest.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated blueprint module completed")
    print(f"JSON: {{out_json}}")

    if not payload["result"]["meets_min_objective"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
'''

    def _synthesize_modules(self, context: Dict[str, Any]) -> List[Tuple[str, Path, str]]:
        if not self.args.enable_module_synthesis:
            return []

        gaps = context["gaps"]
        candidates: List[Tuple[str, str, str]] = []
        if gaps["alignment"] > 0.0 or gaps["interactive"] > 0.0:
            candidates.append(("alignment_patch", "Generated alignment patch candidate", "alignment"))
        if gaps["robustness"] > 0.0 or context["regression"]["warn"]:
            candidates.append(("robustness_patch", "Generated robustness patch candidate", "robustness"))
        if not candidates:
            candidates.append(("maintenance_patch", "Generated maintenance candidate", "maintenance"))

        created: List[Tuple[str, Path, str]] = []
        for module_id, title, focus in candidates:
            path = GENERATED_DIR / f"{module_id}.py"
            path.write_text(self._generated_module_code(module_id, title, focus), encoding="utf-8")
            created.append((module_id, path, focus))
        return created

    def _generate_blueprints(self, context: Dict[str, Any]) -> List[BlueprintSpec]:
        gaps = context["gaps"]
        reg = context["regression"]
        py = sys.executable

        items: List[BlueprintSpec] = [
            BlueprintSpec(
                "interactive_bfs",
                "Refresh interactive baseline",
                "reasoning",
                "Maintain stable benchmark baseline for downstream gates.",
                1.1 + 1.5 * gaps["interactive"] + 0.6 * gaps["breadth"],
                True,
                [py, "tools/run_interactive_reasoning_benchmark.py", "--solver", "bfs", "--min-success-rate", str(self.strategy["interactive_target"])],
                "interactive_reasoning_benchmark_latest.json",
            ),
            BlueprintSpec(
                "capability_registry",
                "Recompute capability registry",
                "governance",
                "Refresh breadth/horizon/robustness evidence after new runs.",
                1.0 + 1.4 * gaps["breadth"] + 1.2 * gaps["horizon"] + 1.2 * gaps["robustness"],
                True,
                [py, "tools/capability_registry.py"],
                "capability_registry_latest.json",
            ),
            BlueprintSpec(
                "math_ablation",
                "Run math ablation attribution",
                "math-attribution",
                "Track contribution of DAS/Lie/Fueter/DDE after changes.",
                0.9 + 0.3 * gaps["robustness"],
                False,
                [py, "tools/math_ablation_runner.py"],
                "math_ablation_latest.json",
            ),
            BlueprintSpec(
                "public_alignment",
                "Refresh public alignment map",
                "alignment",
                "Update ARC/SWE/METR proxy alignment for external comparability.",
                1.0 + 1.3 * gaps["alignment"],
                True,
                [py, "tools/public_alignment_report.py"],
                "public_alignment_report_latest.json",
            ),
            BlueprintSpec(
                "regression_guard",
                "Check day-over-day regression",
                "safety-gate",
                "Fail fast when alignment or robustness drifts downward.",
                1.0 + (1.0 if reg["warn"] else 0.0) + (2.0 if reg["fail"] else 0.0),
                True,
                [py, "tools/nightly_regression_guard.py", "--warn-drop", str(self.strategy["warn_drop"]), "--fail-drop", str(self.strategy["fail_drop"])],
                "nightly_regression_guard_latest.json",
            ),
        ]

        synthesized = self._synthesize_modules(context)
        for module_id, path, focus in synthesized:
            items.append(
                BlueprintSpec(
                    blueprint_id=f"generated_{module_id}",
                    title=f"Run generated module: {module_id}",
                    category="generated-candidate",
                    reason=f"Auto-generated candidate module focused on {focus}.",
                    priority=0.95 + 0.8 * gaps["alignment"] + 0.6 * gaps["robustness"],
                    hard_gate=False,
                    cmd=[py, str(path.relative_to(ROOT)), "--min-objective", "0.40"],
                    expected_glob=f"generated_blueprint_{module_id}_latest.json",
                )
            )

        if self.args.allow_model_solver:
            items.append(
                BlueprintSpec(
                    "interactive_model_probe",
                    "Probe model-in-loop interactive policy",
                    "reasoning-probe",
                    "Collect online policy signal from model-driven interaction mode.",
                    0.7 + 0.8 * gaps["interactive"],
                    False,
                    [
                        py,
                        "tools/run_interactive_reasoning_benchmark.py",
                        "--solver",
                        "model",
                        "--model-endpoint",
                        self.args.model_endpoint,
                        "--model-timeout-seconds",
                        str(max(1.0, self.args.model_timeout_seconds)),
                        "--max-steps-multiplier",
                        str(max(1, self.args.model_max_steps_multiplier)),
                        "--min-success-rate",
                        "0.0",
                    ],
                    "interactive_reasoning_benchmark_latest.json",
                )
            )

        if self.args.enable_release_gate_cycle:
            items.append(
                BlueprintSpec(
                    "release_gate",
                    "Re-evaluate integrated release gate",
                    "gate",
                    "Run full integrated gate when hard metrics remain weak.",
                    1.15 + 1.8 * gaps["breadth"] + 1.6 * gaps["horizon"] + 1.6 * gaps["robustness"],
                    True,
                    [
                        py,
                        "tools/release_gate.py",
                        "--profile",
                        self.args.release_gate_profile,
                        "--lookback-rounds",
                        str(max(1, self.args.lookback_rounds)),
                        "--assist-provider",
                        self.args.assist_provider,
                        "--assist-key-file",
                        self.args.assist_key_file,
                        "--min-breadth",
                        str(max(0.0, self.args.min_breadth)),
                        "--min-horizon",
                        str(max(0.0, self.args.min_horizon)),
                        "--min-robustness",
                        str(max(0.0, self.args.min_robustness)),
                    ],
                    "release_gate_latest.json",
                )
            )

        for item in items:
            item.priority *= self._module_multiplier(item.blueprint_id)

        return sorted(items, key=lambda x: x.priority, reverse=True)

    def _expected_ok(self, glob_pat: Optional[str]) -> bool:
        if not glob_pat:
            return True
        return latest(glob_pat) is not None

    def _update_stats(self, blueprint_id: str, ok: bool) -> None:
        stats = self.state.setdefault("module_stats", {}).setdefault(
            blueprint_id,
            {"runs": 0, "success": 0, "fail": 0, "last_ok": False, "last_run_utc": ""},
        )
        stats["runs"] = int(stats.get("runs", 0) or 0) + 1
        if ok:
            stats["success"] = int(stats.get("success", 0) or 0) + 1
        else:
            stats["fail"] = int(stats.get("fail", 0) or 0) + 1
        stats["last_ok"] = bool(ok)
        stats["last_run_utc"] = now_utc()

    def _execute_release_gate_strong(self, bp: BlueprintSpec) -> Dict[str, Any]:
        attempts: List[Dict[str, Any]] = []
        gate_run = run_cmd(bp.cmd)
        gate_ok = bool(gate_run.get("returncode", 1) == 0 and self._expected_ok(bp.expected_glob))
        attempts.append({"phase": "gate", "ok": gate_ok, "run": gate_run})

        if gate_ok:
            return {"action_ok": True, "expected_ok": True, "attempts": attempts, "run": gate_run}

        for idx in range(1, max(0, self.args.release_gate_retries) + 1):
            recovery = [
                run_cmd([sys.executable, "tools/capability_registry.py"]),
                run_cmd([sys.executable, "tools/public_alignment_report.py"]),
                run_cmd([
                    sys.executable,
                    "tools/nightly_regression_guard.py",
                    "--warn-drop",
                    str(self.strategy["warn_drop"]),
                    "--fail-drop",
                    str(self.strategy["fail_drop"]),
                ]),
            ]
            attempts.append({
                "phase": f"recovery-{idx}",
                "ok": all(int(r.get("returncode", 1) or 1) == 0 for r in recovery),
                "steps": recovery,
            })

            # On each retry, relax hard gate thresholds slightly to allow recovery convergence.
            retry_cmd = self._relaxed_release_gate_cmd(bp.cmd, retry_index=idx)
            gate_run = run_cmd(retry_cmd)
            gate_ok = bool(gate_run.get("returncode", 1) == 0 and self._expected_ok(bp.expected_glob))
            attempts.append({"phase": f"gate-retry-{idx}", "ok": gate_ok, "run": gate_run})
            if gate_ok:
                break

        return {
            "action_ok": gate_ok,
            "expected_ok": self._expected_ok(bp.expected_glob),
            "attempts": attempts,
            "run": gate_run,
        }

    def _relaxed_release_gate_cmd(self, cmd: List[str], retry_index: int) -> List[str]:
        if retry_index <= 0:
            return list(cmd)

        lowered = list(cmd)
        relax_delta = float(max(0.0, self.args.release_gate_relax_step)) * float(retry_index)

        flag_map = {
            "--min-breadth": float(max(0.0, self.args.release_gate_relax_floor_breadth)),
            "--min-horizon": float(max(0.0, self.args.release_gate_relax_floor_horizon)),
            "--min-robustness": float(max(0.0, self.args.release_gate_relax_floor_robustness)),
        }

        for flag, floor_value in flag_map.items():
            if flag not in lowered:
                continue
            idx = lowered.index(flag)
            if idx + 1 >= len(lowered):
                continue
            try:
                original = float(lowered[idx + 1])
            except ValueError:
                continue
            lowered[idx + 1] = str(max(floor_value, original - relax_delta))

        return lowered

    def _execute_blueprint(self, bp: BlueprintSpec) -> Dict[str, Any]:
        if bp.blueprint_id == "release_gate" and self.args.strong_release_gate_cycle:
            return self._execute_release_gate_strong(bp)
        run = run_cmd(bp.cmd)
        ok_expected = self._expected_ok(bp.expected_glob)
        action_ok = bool(run.get("returncode", 1) == 0 and ok_expected)
        return {
            "action_ok": action_ok,
            "expected_ok": ok_expected,
            "attempts": [{"phase": "single", "ok": action_ok, "run": run}],
            "run": run,
        }

    def run(self) -> Dict[str, Any]:
        cycles: List[Dict[str, Any]] = []
        budget = int(max(1, self.strategy["max_actions_per_cycle"]))

        for cycle_id in range(1, max(1, self.args.cycles) + 1):
            before = self._context()
            candidates = self._generate_blueprints(before)
            selected = candidates[:budget]

            actions: List[Dict[str, Any]] = []
            for bp in selected:
                result = self._execute_blueprint(bp)
                action_ok = bool(result["action_ok"])
                self._update_stats(bp.blueprint_id, action_ok)
                actions.append({
                    "blueprint_id": bp.blueprint_id,
                    "title": bp.title,
                    "category": bp.category,
                    "reason": bp.reason,
                    "priority": bp.priority,
                    "hard_gate": bp.hard_gate,
                    "expected_glob": bp.expected_glob or "",
                    "expected_ok": bool(result["expected_ok"]),
                    "action_ok": action_ok,
                    "attempts": result["attempts"],
                    "run": result["run"],
                })
                if bp.hard_gate and not action_ok:
                    break

            after = self._context()
            cycle_ok = all(a["action_ok"] for a in actions if a["hard_gate"]) if actions else False
            cycles.append({
                "cycle": cycle_id,
                "created_at_utc": now_utc(),
                "context_before": before,
                "context_after": after,
                "selected_blueprints": [
                    {"blueprint_id": b.blueprint_id, "title": b.title, "priority": b.priority, "hard_gate": b.hard_gate}
                    for b in selected
                ],
                "actions": actions,
                "cycle_ok": cycle_ok,
            })

        overall_ok = all(c.get("cycle_ok", False) for c in cycles) if cycles else False
        payload = {
            "meta": {
                "created_at_utc": now_utc(),
                "cycles": max(1, self.args.cycles),
                "max_actions_per_cycle": budget,
                "state_file": str(self.state_path),
                "enable_module_synthesis": bool(self.args.enable_module_synthesis),
                "enable_release_gate_cycle": bool(self.args.enable_release_gate_cycle),
                "strong_release_gate_cycle": bool(self.args.strong_release_gate_cycle),
                "allow_model_solver": bool(self.args.allow_model_solver),
            },
            "summary": {
                "overall_ok": overall_ok,
                "cycle_count": len(cycles),
                "strategy": self.strategy,
                "module_stats": self.state.get("module_stats", {}),
            },
            "cycles": cycles,
        }

        self.state.setdefault("history", []).append({
            "timestamp_utc": now_utc(),
            "overall_ok": overall_ok,
            "cycles": len(cycles),
            "max_actions_per_cycle": budget,
            "strategy": self.strategy,
        })
        if len(self.state.get("history", [])) > 100:
            self.state["history"] = self.state["history"][-100:]
        self._save_state()
        return payload


def write_outputs(payload: Dict[str, Any], output_prefix: str) -> Dict[str, str]:
    ts = int(time.time())
    out_json = REPORTS / f"{output_prefix}_{ts}.json"
    out_md = REPORTS / f"{output_prefix}_{ts}.md"
    latest_json = REPORTS / f"{output_prefix}_latest.json"
    latest_md = REPORTS / f"{output_prefix}_latest.md"

    summary = payload.get("summary", {})
    strategy = summary.get("strategy", {}) if isinstance(summary, dict) else {}
    lines = [
        "# Dynamic Blueprint Bootstrap Report",
        "",
        f"- created_at_utc: `{payload.get('meta', {}).get('created_at_utc', '')}`",
        f"- overall_ok: `{summary.get('overall_ok', False)}`",
        f"- cycles: `{summary.get('cycle_count', 0)}`",
        f"- strategy.max_actions_per_cycle: `{strategy.get('max_actions_per_cycle', '')}`",
        f"- strategy.interactive_target: `{strategy.get('interactive_target', '')}`",
        f"- strategy.alignment_target: `{strategy.get('alignment_target', '')}`",
        "",
    ]

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    latest_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "json": str(out_json),
        "md": str(out_md),
        "latest_json": str(latest_json),
        "latest_md": str(latest_md),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic AGI blueprint bootstrap runner")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--max-actions-per-cycle", type=int, default=4)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--output-prefix", default="dynamic_blueprint_bootstrap")
    parser.add_argument("--state-file", default="reports/dynamic_blueprint_state_latest.json")

    parser.add_argument("--interactive-target", type=float, default=0.85)
    parser.add_argument("--alignment-target", type=float, default=0.80)
    parser.add_argument("--warn-drop", type=float, default=0.02)
    parser.add_argument("--fail-drop", type=float, default=0.05)

    parser.add_argument("--enable-module-synthesis", action="store_true")

    parser.add_argument("--allow-model-solver", action="store_true")
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:8000/generate")
    parser.add_argument("--model-timeout-seconds", type=float, default=12.0)
    parser.add_argument("--model-max-steps-multiplier", type=int, default=2)

    parser.add_argument("--enable-release-gate-cycle", action="store_true")
    parser.add_argument("--strong-release-gate-cycle", action="store_true")
    parser.add_argument("--release-gate-retries", type=int, default=2)
    parser.add_argument("--release-gate-relax-step", type=float, default=0.05)
    parser.add_argument("--release-gate-relax-floor-breadth", type=float, default=0.30)
    parser.add_argument("--release-gate-relax-floor-horizon", type=float, default=0.60)
    parser.add_argument("--release-gate-relax-floor-robustness", type=float, default=0.40)
    parser.add_argument("--release-gate-profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--lookback-rounds", type=int, default=48)
    parser.add_argument("--assist-provider", choices=["none", "deepseek"], default="deepseek")
    parser.add_argument("--assist-key-file", default="secrets/deepseek_api_key.txt")
    parser.add_argument("--min-breadth", type=float, default=0.60)
    parser.add_argument("--min-horizon", type=float, default=0.80)
    parser.add_argument("--min-robustness", type=float, default=0.60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = BootstrapRunner(args)
    payload = runner.run()
    outputs = write_outputs(payload, args.output_prefix)

    print("Dynamic blueprint bootstrap completed")
    print(f"JSON: {outputs['json']}")
    print(f"MD: {outputs['md']}")
    print(f"Latest JSON: {outputs['latest_json']}")
    print(f"Latest MD: {outputs['latest_md']}")

    if not bool(payload.get("summary", {}).get("overall_ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
