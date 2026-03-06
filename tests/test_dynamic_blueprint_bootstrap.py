"""Minimal unit tests for dynamic blueprint bootstrap orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

import tools.dynamic_blueprint_bootstrap as dbb


def _make_args(state_file: str, **overrides):
    base = dict(
        cycles=1,
        max_actions_per_cycle=4,
        history_window=8,
        output_prefix="dynamic_blueprint_bootstrap",
        state_file=state_file,
        interactive_target=0.85,
        alignment_target=0.80,
        warn_drop=0.02,
        fail_drop=0.05,
        enable_module_synthesis=False,
        allow_model_solver=False,
        model_endpoint="http://127.0.0.1:8000/generate",
        model_timeout_seconds=12.0,
        model_max_steps_multiplier=2,
        enable_release_gate_cycle=False,
        strong_release_gate_cycle=False,
        release_gate_retries=2,
        release_gate_relax_step=0.05,
        release_gate_relax_floor_breadth=0.30,
        release_gate_relax_floor_horizon=0.60,
        release_gate_relax_floor_robustness=0.40,
        release_gate_profile="quick",
        lookback_rounds=48,
        assist_provider="deepseek",
        assist_key_file="secrets/deepseek_api_key.txt",
        min_breadth=0.60,
        min_horizon=0.80,
        min_robustness=0.60,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _isolate_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dbb, "ROOT", tmp_path)
    monkeypatch.setattr(dbb, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(dbb, "GENERATED_DIR", tmp_path / "tools" / "generated_blueprints")


def test_strategy_learning_relaxes_thresholds_on_fail_streak(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_paths(monkeypatch, tmp_path)
    state_path = tmp_path / "reports" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "meta": {"version": 2},
                "history": [
                    {"overall_ok": False},
                    {"overall_ok": False},
                    {"overall_ok": False},
                ],
                "module_stats": {},
                "strategy": {},
            }
        ),
        encoding="utf-8",
    )

    args = _make_args("reports/state.json", max_actions_per_cycle=3)
    runner = dbb.BootstrapRunner(args)

    assert runner.strategy["max_actions_per_cycle"] == 5
    assert runner.strategy["interactive_target"] == pytest.approx(0.82)
    assert runner.strategy["alignment_target"] == pytest.approx(0.77)
    assert runner.strategy["fail_streak"] == 3


def test_blueprint_generation_includes_release_gate_and_generated_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_paths(monkeypatch, tmp_path)
    args = _make_args(
        "reports/state.json",
        enable_module_synthesis=True,
        enable_release_gate_cycle=True,
    )
    runner = dbb.BootstrapRunner(args)

    context = {
        "scores": {
            "breadth": 0.2,
            "horizon": 0.4,
            "robustness": 0.2,
            "alignment_overall": 0.4,
            "interactive_success": 0.5,
        },
        "gaps": {
            "breadth": 0.4,
            "horizon": 0.4,
            "robustness": 0.4,
            "alignment": 0.4,
            "interactive": 0.35,
        },
        "regression": {"warn": True, "fail": False},
    }

    blueprints = runner._generate_blueprints(context)
    ids = {bp.blueprint_id for bp in blueprints}

    assert "release_gate" in ids
    assert any(bp.blueprint_id.startswith("generated_") for bp in blueprints)
    assert (tmp_path / "tools" / "generated_blueprints").exists()


def test_strong_release_gate_retry_recovers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_paths(monkeypatch, tmp_path)
    args = _make_args(
        "reports/state.json",
        strong_release_gate_cycle=True,
        release_gate_retries=2,
    )
    runner = dbb.BootstrapRunner(args)

    # First gate attempt fails, then recovery succeeds, then retry succeeds.
    results = [
        {"returncode": 1, "stdout_tail": "gate failed", "stderr_tail": ""},
        {"returncode": 0, "stdout_tail": "cap ok", "stderr_tail": ""},
        {"returncode": 0, "stdout_tail": "align ok", "stderr_tail": ""},
        {"returncode": 0, "stdout_tail": "reg ok", "stderr_tail": ""},
        {"returncode": 0, "stdout_tail": "gate retry ok", "stderr_tail": ""},
    ]

    def _fake_run_cmd(_cmd):
        return results.pop(0)

    monkeypatch.setattr(dbb, "run_cmd", _fake_run_cmd)
    monkeypatch.setattr(runner, "_expected_ok", lambda _glob: True)

    bp = dbb.BlueprintSpec(
        blueprint_id="release_gate",
        title="release gate",
        category="gate",
        reason="test",
        priority=1.0,
        hard_gate=True,
        cmd=["python", "tools/release_gate.py"],
        expected_glob="release_gate_latest.json",
    )

    result = runner._execute_blueprint(bp)

    assert result["action_ok"] is True
    phases = [a["phase"] for a in result["attempts"]]
    assert phases == ["gate", "recovery-1", "gate-retry-1"]


def test_strong_release_gate_auto_relax_then_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _isolate_paths(monkeypatch, tmp_path)
    args = _make_args(
        "reports/state.json",
        strong_release_gate_cycle=True,
        release_gate_retries=1,
        release_gate_relax_step=0.20,
        release_gate_relax_floor_breadth=0.70,
        release_gate_relax_floor_horizon=0.70,
        release_gate_relax_floor_robustness=0.70,
    )
    runner = dbb.BootstrapRunner(args)
    monkeypatch.setattr(runner, "_expected_ok", lambda _glob: True)

    def _fake_run_cmd(cmd):
        cmd_s = " ".join(cmd)
        if "tools/capability_registry.py" in cmd_s:
            return {"returncode": 0, "stdout_tail": "cap ok", "stderr_tail": "", "cmd": cmd}
        if "tools/public_alignment_report.py" in cmd_s:
            return {"returncode": 0, "stdout_tail": "align ok", "stderr_tail": "", "cmd": cmd}
        if "tools/nightly_regression_guard.py" in cmd_s:
            return {"returncode": 0, "stdout_tail": "reg ok", "stderr_tail": "", "cmd": cmd}

        # Gate fails at 0.99 and succeeds after retry relaxation to 0.79.
        breadth = None
        if "--min-breadth" in cmd:
            breadth = float(cmd[cmd.index("--min-breadth") + 1])
        rc = 0 if (breadth is not None and breadth <= 0.79) else 1
        return {"returncode": rc, "stdout_tail": f"gate breadth={breadth}", "stderr_tail": "", "cmd": cmd}

    monkeypatch.setattr(dbb, "run_cmd", _fake_run_cmd)

    bp = dbb.BlueprintSpec(
        blueprint_id="release_gate",
        title="release gate",
        category="gate",
        reason="test auto relax",
        priority=1.0,
        hard_gate=True,
        cmd=[
            "python",
            "tools/release_gate.py",
            "--min-breadth",
            "0.99",
            "--min-horizon",
            "0.99",
            "--min-robustness",
            "0.99",
        ],
        expected_glob="release_gate_latest.json",
    )

    result = runner._execute_blueprint(bp)

    assert result["action_ok"] is True
    phases = [a["phase"] for a in result["attempts"]]
    assert phases == ["gate", "recovery-1", "gate-retry-1"]
    retry_cmd = result["attempts"][-1]["run"]["cmd"]
    assert retry_cmd[retry_cmd.index("--min-breadth") + 1] == "0.79"
