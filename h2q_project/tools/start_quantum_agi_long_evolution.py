#!/usr/bin/env python3
"""Start a headless long-running quantum AGI software-simulation evolution instance.

This runner integrates:
- STQ quantum witness evolution simulation.
- AGI capability checks from local evolution modules.
- Local knowledge acquisition/compression loops.
- Survival daemon heartbeat and recovery callbacks.

Outputs:
- quantum_agi_long_state.json
- quantum_agi_cycles.jsonl
- quantum_agi_long_report.json
- quantum_agi_long_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from STQ_QuantumSimulator import STQ_QuantumSimulator  # noqa: E402
from h2q_project.h2q.agi.evolution_24h import (  # noqa: E402
    CapabilityTester,
    FractalCompressor,
    KnowledgeAcquirer,
)
from h2q_project.h2q.agi.survival_daemon import (  # noqa: E402
    SurvivalConfig,
    create_survival_daemon,
)


@dataclass
class RunnerConfig:
    hours: float
    cycle_seconds: int
    time_points: int
    capability_check_every: int
    max_cycles: int
    max_knowledge_items: int
    compression_threshold: float
    mass_kg: float
    distance_m: float
    gamma_base: float
    lambda_threshold: float
    formula_mode: str
    resume: bool
    output_dir: Path


class QuantumAGILongEvolutionRunner:
    def __init__(self, cfg: RunnerConfig, *, china_mode: Optional[bool]) -> None:
        self.cfg = cfg
        self.start_ts = time.time()
        self.stop_requested = False

        self.quantum_sim = STQ_QuantumSimulator(mass_kg=cfg.mass_kg, distance_m=cfg.distance_m)
        self.tester = CapabilityTester()
        self.acquirer = KnowledgeAcquirer(china_mode=china_mode)
        self.compressor = FractalCompressor(compression_ratio=0.55)

        self.cycles: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, Any] = {}
        self.latest_capability_score = 0.0
        self.latest_composite_score = 0.0
        self.resumed = False
        self.resumed_from_cycle = 0
        self.resumed_at: Optional[str] = None

        self.curriculum = [
            "quantum_mechanics",
            "topological_quantum_computing",
            "machine_learning",
            "mathematics",
            "computer_science",
            "physics",
            "information_theory",
            "causal_inference",
        ]

        self.output_dir = cfg.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "quantum_agi_long_state.json"
        self.cycles_path = self.output_dir / "quantum_agi_cycles.jsonl"
        self.report_json_path = self.output_dir / "quantum_agi_long_report.json"
        self.report_md_path = self.output_dir / "quantum_agi_long_report.md"

        self.baseline_snapshot = self._load_local_baselines()
        self._restore_if_requested()

    def request_stop(self) -> None:
        self.stop_requested = True

    def _elapsed_hours(self) -> float:
        return (time.time() - self.start_ts) / 3600.0

    def _load_local_baselines(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}

        paths = {
            "paper2206_report": PROJECT_ROOT / "h2q_project/reports/paper2206_ray_structure_space/paper2206_argument_space_report.json",
            "quantum_equivalence": PROJECT_ROOT / "quantum_equivalence_report.json",
            "agi_benchmark": PROJECT_ROOT / "AGI_BENCHMARK_REPORT.json",
            "agi_eval": PROJECT_ROOT / "agi_evaluation_results.json",
        }

        for name, path in paths.items():
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            if name == "paper2206_report":
                snapshot[name] = {
                    "overall_passed": bool(data.get("overall_passed", False)),
                    "checks": data.get("checks", {}),
                }
            elif name == "quantum_equivalence":
                h2q = data.get("h2q_results", {})
                ghz = h2q.get("GHZ_State_Fidelity", {})
                snapshot[name] = {
                    "timestamp": data.get("timestamp"),
                    "bv": h2q.get("Bernstein_Vazirani"),
                    "ghz_fidelity_mean": float(np.mean(list(ghz.values()))) if ghz else 0.0,
                }
            elif name == "agi_benchmark":
                snapshot[name] = {
                    "overall_score": float(data.get("overall_score", 0.0)),
                    "pass_rate": float(data.get("overall_pass_rate", 0.0)),
                    "verdict": data.get("superiority_verdict", "unknown"),
                }
            elif name == "agi_eval":
                scores = data.get("scores", {})
                snapshot[name] = {
                    "overall_score": float(scores.get("overall_score", 0.0)),
                    "consciousness_score": float(scores.get("consciousness_score", 0.0)),
                    "adaptability_score": float(scores.get("adaptability_score", 0.0)),
                }

        return snapshot

    def _load_existing_state(self) -> Optional[Dict[str, Any]]:
        if not self.state_path.exists():
            return None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _load_existing_cycles(self) -> List[Dict[str, Any]]:
        if not self.cycles_path.exists():
            return []

        records: List[Dict[str, Any]] = []
        try:
            with open(self.cycles_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        except Exception:
            return []

        return records

    def _parse_iso_ts(self, raw: Any) -> Optional[float]:
        if not isinstance(raw, str):
            return None
        try:
            return datetime.fromisoformat(raw).timestamp()
        except Exception:
            return None

    def _restore_knowledge_placeholders(self, count: int) -> None:
        restore_count = max(0, min(count, self.cfg.max_knowledge_items))
        for idx in range(restore_count):
            key = f"restored_knowledge_{idx:06d}"
            self.knowledge_base[key] = {"restored": True}

    def _restore_if_requested(self) -> None:
        if not self.cfg.resume:
            return

        state = self._load_existing_state()
        restored_cycles = self._load_existing_cycles()

        if not state and not restored_cycles:
            return

        if restored_cycles:
            self.cycles = restored_cycles
            self.resumed_from_cycle = len(restored_cycles)

        if state:
            started_ts = self._parse_iso_ts(state.get("started_at"))
            if started_ts is not None:
                self.start_ts = started_ts

            try:
                self.latest_capability_score = float(state.get("latest_capability_score", self.latest_capability_score))
            except Exception:
                pass

            try:
                self.latest_composite_score = float(state.get("latest_composite_score", self.latest_composite_score))
            except Exception:
                pass

            try:
                knowledge_count = int(state.get("knowledge_count", 0))
            except Exception:
                knowledge_count = 0
            self._restore_knowledge_placeholders(knowledge_count)

            try:
                self.acquirer.acquired_count = max(
                    self.acquirer.acquired_count,
                    int(state.get("acquired_count", 0)),
                )
            except Exception:
                pass

            try:
                self.acquirer.failed_count = max(
                    self.acquirer.failed_count,
                    int(state.get("failed_count", 0)),
                )
            except Exception:
                pass

            if self.resumed_from_cycle == 0:
                try:
                    self.resumed_from_cycle = max(0, int(state.get("cycle_count", 0)))
                except Exception:
                    pass

        self.resumed = True
        self.resumed_at = datetime.now().isoformat()

    def _compute_ci(self, values: List[float], alpha: float = 0.05) -> Tuple[float, float, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return 0.0, 0.0, 0.0
        mean = float(np.mean(arr))
        if arr.size == 1:
            return mean, mean, mean

        rng = np.random.default_rng(20260406 + arr.size)
        samples = np.zeros(1200, dtype=np.float64)
        for i in range(samples.size):
            boot = rng.choice(arr, size=arr.size, replace=True)
            samples[i] = float(np.mean(boot))

        lo = float(np.quantile(samples, alpha / 2.0))
        hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
        return mean, lo, hi

    def _knowledge_step(self, cycle_idx: int) -> Dict[str, Any]:
        topic = self.curriculum[cycle_idx % len(self.curriculum)]
        acquired = self.acquirer.fetch_summary(topic)
        added = False

        if acquired:
            key = f"{topic}_{cycle_idx:06d}"
            self.knowledge_base[key] = acquired
            added = True

        if len(self.knowledge_base) > int(self.cfg.max_knowledge_items * self.cfg.compression_threshold):
            compressed: Dict[str, Any] = {}
            for key, val in self.knowledge_base.items():
                compressed[key] = self.compressor.compress(val) if isinstance(val, dict) else val
            self.knowledge_base = compressed

        if len(self.knowledge_base) > self.cfg.max_knowledge_items:
            keys = sorted(self.knowledge_base.keys())
            drop = len(self.knowledge_base) - self.cfg.max_knowledge_items
            for key in keys[:drop]:
                self.knowledge_base.pop(key, None)

        return {
            "topic": topic,
            "added": added,
            "knowledge_count": len(self.knowledge_base),
            "acquired_count": self.acquirer.acquired_count,
            "failed_count": self.acquirer.failed_count,
        }

    def _quantum_step(self, cycle_idx: int) -> Dict[str, Any]:
        horizon = 2.0 + 0.03 * float(cycle_idx)
        t = np.linspace(0.0, horizon, self.cfg.time_points)

        gamma = self.cfg.gamma_base * (1.0 + 0.01 * min(cycle_idx, 300))
        lambda_threshold = self.cfg.lambda_threshold * (1.0 + 0.02 * math.sin(cycle_idx / 13.0))

        witness = self.quantum_sim.dual_complex_evolution(
            t,
            gamma_decoherence=gamma,
            Lambda_threshold=lambda_threshold,
            formula_mode=self.cfg.formula_mode,
        )
        w = np.asarray(witness, dtype=np.float64)

        w_min = float(np.min(w))
        w_max = float(np.max(w))
        w_mean = float(np.mean(w))
        w_std = float(np.std(w))
        entanglement_ratio = float(np.mean(w < 0.0))

        return {
            "horizon": horizon,
            "gamma": gamma,
            "lambda_threshold": lambda_threshold,
            "witness_min": w_min,
            "witness_max": w_max,
            "witness_mean": w_mean,
            "witness_std": w_std,
            "entanglement_negative_ratio": entanglement_ratio,
            "entanglement_detected": bool(w_min < 0.0),
        }

    def _capability_step(self, cycle_idx: int) -> Dict[str, Any]:
        do_check = (cycle_idx % max(1, self.cfg.capability_check_every)) == 0
        if not do_check:
            return {
                "checked": False,
                "overall_score": self.latest_capability_score * 100.0,
            }

        result = self.tester.run_comprehensive_test()
        score = float(result.get("overall_score", 0.0))
        self.latest_capability_score = score / 100.0

        return {
            "checked": True,
            "overall_score": score,
            "grade": result.get("grade", "unknown"),
            "tests": {k: float(v.get("score", 0.0)) for k, v in result.get("tests", {}).items()},
        }

    def _composite_score(
        self,
        *,
        quantum: Dict[str, Any],
        capability: Dict[str, Any],
        knowledge_count: int,
    ) -> float:
        witness_quality = float(np.clip(-quantum["witness_min"], 0.0, 1.0))
        witness_stability = 1.0 / (1.0 + max(0.0, quantum["witness_std"]))
        capability_norm = float(np.clip(capability.get("overall_score", 0.0) / 100.0, 0.0, 1.0))
        knowledge_norm = float(np.clip(knowledge_count / 800.0, 0.0, 1.0))

        score = 0.40 * witness_quality + 0.25 * witness_stability + 0.25 * capability_norm + 0.10 * knowledge_norm
        return float(np.clip(score, 0.0, 1.0))

    def _write_cycle(self, record: Dict[str, Any]) -> None:
        with open(self.cycles_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _save_state(self) -> None:
        data = {
            "started_at": datetime.fromtimestamp(self.start_ts).isoformat(),
            "saved_at": datetime.now().isoformat(),
            "elapsed_hours": self._elapsed_hours(),
            "cycle_count": len(self.cycles),
            "latest_capability_score": self.latest_capability_score,
            "latest_composite_score": self.latest_composite_score,
            "knowledge_count": len(self.knowledge_base),
            "acquired_count": self.acquirer.acquired_count,
            "failed_count": self.acquirer.failed_count,
            "resume_enabled": self.cfg.resume,
            "resumed": self.resumed,
            "resumed_from_cycle": self.resumed_from_cycle,
            "resumed_at": self.resumed_at,
        }
        temp_path = self.state_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.state_path)

    def run(self, daemon: Any) -> Dict[str, Any]:
        print("=" * 72)
        print("Quantum AGI long evolution runner started")
        print(f"Output dir: {self.output_dir}")
        print(f"Target duration: {self.cfg.hours:.2f} hours")
        print(f"Resume enabled: {self.cfg.resume}")
        if self.resumed:
            print(f"Resumed from existing history: {self.resumed_from_cycle} cycles")
        print("=" * 72)

        cycle = len(self.cycles)

        while not self.stop_requested:
            elapsed_h = self._elapsed_hours()
            if elapsed_h >= self.cfg.hours:
                break
            if self.cfg.max_cycles > 0 and cycle >= self.cfg.max_cycles:
                break

            cycle += 1
            cycle_started = time.time()

            try:
                knowledge = self._knowledge_step(cycle)
                quantum = self._quantum_step(cycle)
                capability = self._capability_step(cycle)
                composite = self._composite_score(
                    quantum=quantum,
                    capability=capability,
                    knowledge_count=knowledge["knowledge_count"],
                )
                self.latest_composite_score = composite

                record = {
                    "cycle": cycle,
                    "timestamp": datetime.now().isoformat(),
                    "elapsed_hours": elapsed_h,
                    "quantum": quantum,
                    "capability": capability,
                    "knowledge": knowledge,
                    "composite_score": composite,
                }
                self.cycles.append(record)
                self._write_cycle(record)
                self._save_state()

                daemon.report_task_complete()

                print(
                    f"[cycle {cycle:05d}] witness_min={quantum['witness_min']:.4f} "
                    f"cap={capability.get('overall_score', 0.0):.1f}% "
                    f"knowledge={knowledge['knowledge_count']} composite={composite:.4f}"
                )

            except Exception as exc:
                daemon.report_error()
                print(f"[cycle {cycle:05d}] error: {exc}")

            spent = time.time() - cycle_started
            sleep_s = max(1, self.cfg.cycle_seconds - int(spent))
            for _ in range(sleep_s):
                if self.stop_requested:
                    break
                time.sleep(1)

        return self._finalize()

    def _finalize(self) -> Dict[str, Any]:
        witness_min_vals = [float(c["quantum"]["witness_min"]) for c in self.cycles]
        composite_vals = [float(c["composite_score"]) for c in self.cycles]
        ent_ratio_vals = [float(c["quantum"]["entanglement_negative_ratio"]) for c in self.cycles]

        w_mean, w_lo, w_hi = self._compute_ci(witness_min_vals)
        c_mean, c_lo, c_hi = self._compute_ci(composite_vals)
        e_mean, e_lo, e_hi = self._compute_ci(ent_ratio_vals)

        capability_scores = [
            float(c["capability"].get("overall_score", 0.0))
            for c in self.cycles
            if bool(c["capability"].get("checked"))
        ]
        cap_mean, cap_lo, cap_hi = self._compute_ci(capability_scores)

        report = {
            "started_at": datetime.fromtimestamp(self.start_ts).isoformat(),
            "finished_at": datetime.now().isoformat(),
            "elapsed_hours": self._elapsed_hours(),
            "cycles": len(self.cycles),
            "knowledge_count": len(self.knowledge_base),
            "acquired_count": self.acquirer.acquired_count,
            "failed_count": self.acquirer.failed_count,
            "baselines": self.baseline_snapshot,
            "metrics": {
                "witness_min_ci95": {"mean": w_mean, "lower": w_lo, "upper": w_hi},
                "composite_score_ci95": {"mean": c_mean, "lower": c_lo, "upper": c_hi},
                "entanglement_ratio_ci95": {"mean": e_mean, "lower": e_lo, "upper": e_hi},
                "capability_score_ci95": {"mean": cap_mean, "lower": cap_lo, "upper": cap_hi},
            },
            "resume": {
                "enabled": self.cfg.resume,
                "resumed": self.resumed,
                "resumed_from_cycle": self.resumed_from_cycle,
                "resumed_at": self.resumed_at,
            },
            "latest_capability_score": self.latest_capability_score,
            "latest_composite_score": self.latest_composite_score,
        }

        self.report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        md = []
        md.append("# Quantum AGI Long Evolution Report")
        md.append("")
        md.append(f"- started_at: {report['started_at']}")
        md.append(f"- finished_at: {report['finished_at']}")
        md.append(f"- elapsed_hours: {report['elapsed_hours']:.4f}")
        md.append(f"- cycles: {report['cycles']}")
        md.append(f"- knowledge_count: {report['knowledge_count']}")
        md.append("")
        md.append("## CI Metrics (95%)")
        md.append("")
        md.append(
            "- witness_min mean/lower/upper: "
            f"{w_mean:.6f} / {w_lo:.6f} / {w_hi:.6f}"
        )
        md.append(
            "- composite_score mean/lower/upper: "
            f"{c_mean:.6f} / {c_lo:.6f} / {c_hi:.6f}"
        )
        md.append(
            "- entanglement_ratio mean/lower/upper: "
            f"{e_mean:.6f} / {e_lo:.6f} / {e_hi:.6f}"
        )
        md.append(
            "- capability_score mean/lower/upper: "
            f"{cap_mean:.6f} / {cap_lo:.6f} / {cap_hi:.6f}"
        )

        self.report_md_path.write_text("\n".join(md), encoding="utf-8")
        self._save_state()

        return report


def _parse_china_mode(mode: str) -> Optional[bool]:
    mode = mode.strip().lower()
    if mode == "auto":
        return None
    if mode in {"on", "true", "1"}:
        return True
    if mode in {"off", "false", "0"}:
        return False
    return None


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless long-running quantum AGI evolution runner")
    parser.add_argument("--hours", type=float, default=24.0, help="Target runtime in hours")
    parser.add_argument("--cycle-seconds", type=int, default=120, help="Seconds per evolution cycle")
    parser.add_argument("--time-points", type=int, default=220, help="Time samples per quantum simulation")
    parser.add_argument("--capability-check-every", type=int, default=5, help="Run capability test every N cycles")
    parser.add_argument("--max-cycles", type=int, default=0, help="Optional hard cap for cycles (0 means unlimited)")
    parser.add_argument("--max-knowledge-items", type=int, default=2500, help="Knowledge base size limit")
    parser.add_argument("--compression-threshold", type=float, default=0.80, help="Compression threshold ratio")

    parser.add_argument("--mass-kg", type=float, default=1e-14, help="Quantum simulator mass parameter")
    parser.add_argument("--distance-m", type=float, default=35e-6, help="Quantum simulator distance parameter")
    parser.add_argument("--gamma-base", type=float, default=1e-3, help="Base decoherence gamma")
    parser.add_argument("--lambda-threshold", type=float, default=4.0, help="Quantum witness lock threshold")
    parser.add_argument("--formula-mode", choices=["legacy", "aligned"], default="aligned", help="Witness formula mode")
    parser.add_argument("--resume", action="store_true", help="Resume from existing state/cycle files in output dir")

    parser.add_argument("--china-mode", choices=["auto", "on", "off"], default="auto", help="Knowledge source mode")
    parser.add_argument("--output-dir", default="h2q_project/reports/quantum_agi_long_evolution", help="Output directory (or base dir when separate-run-dir=on)")
    parser.add_argument("--separate-run-dir", choices=["on", "off"], default="off", help="Use a timestamped subdirectory under output-dir")
    parser.add_argument("--run-name", default="", help="Optional run directory name when separate-run-dir=on")
    parser.add_argument("--latest-link-name", default="latest", help="Symlink name under output-dir base that points to current run dir")

    parser.add_argument("--daemon-heartbeat", type=int, default=30, help="Survival daemon heartbeat interval")
    parser.add_argument("--daemon-timeout", type=int, default=150, help="Survival daemon max no-heartbeat seconds")
    parser.add_argument("--daemon-restart-cooldown", type=int, default=60, help="Survival daemon restart cooldown")

    return parser.parse_args(argv)


def _resolve_output_dir(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    base_output_dir = Path(args.output_dir)
    if not base_output_dir.is_absolute():
        base_output_dir = (PROJECT_ROOT / base_output_dir).resolve()
    else:
        base_output_dir = base_output_dir.resolve()

    if args.separate_run_dir == "off":
        return base_output_dir, None

    run_name = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return (base_output_dir / run_name).resolve(), base_output_dir


def _update_latest_symlink(base_output_dir: Path, output_dir: Path, link_name: str) -> None:
    name = link_name.strip()
    if not name:
        return

    base_output_dir.mkdir(parents=True, exist_ok=True)
    link_path = base_output_dir / name

    try:
        if link_path.is_symlink():
            link_path.unlink()
        elif link_path.exists():
            # Avoid deleting user-created directories/files.
            print(f"Warning: skip latest link update because path exists and is not symlink: {link_path}")
            return

        link_path.symlink_to(output_dir, target_is_directory=True)
    except Exception as exc:
        print(f"Warning: failed to update latest link {link_path}: {exc}")


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    output_dir, base_output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    if base_output_dir is not None:
        _update_latest_symlink(base_output_dir, output_dir, args.latest_link_name)

    cfg = RunnerConfig(
        hours=max(0.01, args.hours),
        cycle_seconds=max(5, args.cycle_seconds),
        time_points=max(40, args.time_points),
        capability_check_every=max(1, args.capability_check_every),
        max_cycles=max(0, args.max_cycles),
        max_knowledge_items=max(100, args.max_knowledge_items),
        compression_threshold=float(np.clip(args.compression_threshold, 0.2, 0.98)),
        mass_kg=max(1e-20, args.mass_kg),
        distance_m=max(1e-12, args.distance_m),
        gamma_base=max(1e-8, args.gamma_base),
        lambda_threshold=max(1e-6, args.lambda_threshold),
        formula_mode=args.formula_mode,
        resume=bool(args.resume),
        output_dir=output_dir,
    )

    runner = QuantumAGILongEvolutionRunner(cfg, china_mode=_parse_china_mode(args.china_mode))

    daemon_cfg = SurvivalConfig(
        heartbeat_interval=max(5, args.daemon_heartbeat),
        max_no_heartbeat=max(30, args.daemon_timeout),
        restart_cooldown=max(10, args.daemon_restart_cooldown),
        state_file=str((output_dir / "quantum_agi_survival_state.json").resolve()),
        heartbeat_file=str((output_dir / "quantum_agi_heartbeat.json").resolve()),
        log_file=str((output_dir / "quantum_agi_survival.log").resolve()),
    )
    daemon = create_survival_daemon(work_dir=str(PROJECT_ROOT), config=daemon_cfg)

    def _capability_cb() -> float:
        return float(100.0 * runner.latest_composite_score)

    def _restart_cb() -> None:
        runner.quantum_sim = STQ_QuantumSimulator(mass_kg=cfg.mass_kg, distance_m=cfg.distance_m)

    daemon.set_capability_callback(_capability_cb)
    daemon.set_restart_callback(_restart_cb)

    def _signal_handler(signum: int, _frame: Any) -> None:
        print(f"\nReceived signal {signum}, requesting graceful stop...")
        runner.request_stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    daemon.start()
    exit_code = 0

    try:
        report = runner.run(daemon)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except Exception as exc:
        daemon.report_error()
        print(f"Fatal error: {exc}")
        exit_code = 2
    finally:
        daemon.stop()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
