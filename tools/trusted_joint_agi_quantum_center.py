import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from das_experimental_validator import DASExperimentalValidator
from tools.dual_conjugate_joint_tuning import evaluate_dual_conjugate_modes
from tools.industrial_realtime_codec import benchmark_on_files


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 256)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def run_das_validation(monte_carlo_samples: int) -> Dict[str, object]:
    validator = DASExperimentalValidator(monte_carlo_samples=monte_carlo_samples, seed=42, precision_eps=1e-18)
    report = validator.build_statistical_report()
    return {
        "report": report,
        "key": {
            "decision_grade_ready": bool(report["verdict"].get("decision_grade_ready", False)),
            "physics_ready": bool(report["verdict"].get("physics_ready", False)),
            "isomorphic_confidence_score": float(report["confidence"].get("isomorphic_confidence_score", 0.0)),
            "dual_conjugate_aligned_pass": bool(report["verdict"].get("dual_conjugate_aligned_pass", False)),
        },
    }


def run_dual_conjugate_tuning() -> Dict[str, object]:
    payload = evaluate_dual_conjugate_modes()
    summary = payload["summary"]
    return {
        "report": payload,
        "key": {
            "best_mode": summary.get("best_mode"),
            "aligned_corr": float(summary.get("aligned_corr", 0.0)),
            "aligned_mae": float(summary.get("aligned_mae", 1e9)),
        },
    }


def run_rsa_parallel_cv(k_folds: int) -> Dict[str, object]:
    # Lazy import avoids hard dependency until this stage is enabled.
    from tools.rsa_parallel_cv_no_timefold import cross_validate_parallel_advantage

    payload = cross_validate_parallel_advantage(k_folds=k_folds)
    verdict = payload["verdict"]
    return {
        "report": payload,
        "key": {
            "mean_parallel_speedup": float(verdict.get("mean_parallel_speedup", 0.0)),
            "has_final_usability": bool(verdict.get("has_final_usability", False)),
            "has_real_quantum_advantage": bool(verdict.get("has_real_quantum_advantage", False)),
            "quantum_advantage_case_count": int(verdict.get("quantum_advantage_case_count", 0)),
            "total_case_count": int(verdict.get("total_case_count", 0)),
        },
    }


def run_codec_regression(inputs: List[Path], out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = benchmark_on_files(inputs, out_dir)
    summary = payload["summary"]
    return {
        "report": payload,
        "key": {
            "file_count": int(summary.get("file_count", 0)),
            "mean_ratio": float(summary.get("mean_ratio", 0.0)),
            "mean_compress_mb_s": float(summary.get("mean_compress_mb_s", 0.0)),
            "mean_decompress_mb_s": float(summary.get("mean_decompress_mb_s", 0.0)),
            "all_checksum_match": bool(summary.get("all_checksum_match", False)),
        },
    }


def aggregate_verdict(stages: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    das = stages["das"]["key"]
    dual = stages["dual"]["key"]
    rsa = stages.get("rsa", {}).get("key", {})
    codec = stages["codec"]["key"]

    gates = {
        "das_decision_ready": bool(das.get("decision_grade_ready", False)),
        "dual_aligned_consistent": bool(
            dual.get("best_mode") == "aligned"
            and float(dual.get("aligned_corr", 0.0)) >= 0.80
            and float(dual.get("aligned_mae", 1e9)) <= 0.08
        ),
        "codec_integrity": bool(codec.get("all_checksum_match", False)),
        "rsa_parallel_observed": bool(float(rsa.get("mean_parallel_speedup", 0.0)) > 0.0) if rsa else True,
    }

    score_parts = [
        min(1.0, max(0.0, float(das.get("isomorphic_confidence_score", 0.0)))),
        min(1.0, max(0.0, float(dual.get("aligned_corr", 0.0)))),
        1.0 if codec.get("all_checksum_match", False) else 0.0,
    ]
    if rsa:
        score_parts.append(min(1.0, max(0.0, float(rsa.get("mean_parallel_speedup", 0.0)) / 1.2)))

    trust_score = float(sum(score_parts) / max(len(score_parts), 1))
    trusted_ready = bool(all(gates.values()) and trust_score >= 0.75)

    return {
        "gates": gates,
        "trust_score": trust_score,
        "trusted_ready": trusted_ready,
        "note": "This is a trusted orchestration readiness score over real implemented modules, not a claim of general AGI attainment.",
    }


def write_report_markdown(output_json: Path, payload: Dict[str, object]) -> Path:
    agg = payload["aggregate"]
    das = payload["stages"]["das"]["key"]
    dual = payload["stages"]["dual"]["key"]
    codec = payload["stages"]["codec"]["key"]
    rsa = payload["stages"].get("rsa", {}).get("key")

    lines = [
        "# 可信联合AGI模拟综合量子化系统中心报告",
        "",
        "## 1. 中心程序定义",
        "",
        "本中心程序是对仓库中真实实现模块的级联联调编排，不替换原始算法实现。",
        "目标是构建一个可复验的“综合软件体”运行中枢：统一调度、统一判据、统一可信汇总。",
        "",
        "## 2. 级联阶段",
        "",
        "1. DAS 主验证链（含 aligned 主判据）。",
        "2. 双复数共轭联调（门级见证者对照）。",
        "3. RSA 去时间折叠并行交叉验证（可选阶段）。",
        "4. 工业编解码回归（完整性与吞吐基准）。",
        "",
        "## 3. 关键结果",
        "",
        f"- DAS decision ready: `{das['decision_grade_ready']}`",
        f"- DAS confidence: `{das['isomorphic_confidence_score']:.4f}`",
        f"- Dual best mode: `{dual['best_mode']}`",
        f"- Dual aligned corr/mae: `{dual['aligned_corr']:.4f}` / `{dual['aligned_mae']:.4f}`",
        f"- Codec checksum pass: `{codec['all_checksum_match']}`",
        f"- Codec mean ratio: `{codec['mean_ratio']:.3f}x`",
    ]

    if rsa is not None:
        lines.extend(
            [
                f"- RSA mean parallel speedup: `{rsa['mean_parallel_speedup']:.3f}x`",
                f"- RSA real quantum advantage: `{rsa['has_real_quantum_advantage']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 4. 可信聚合判定",
            "",
            f"- Trust score: `{agg['trust_score']:.4f}`",
            f"- Trusted ready: `{agg['trusted_ready']}`",
            f"- Gate detail: `{agg['gates']}`",
            "",
            "## 5. 附件",
            "",
            f"- 结构化数据：`{output_json}`",
            f"- 说明：`{agg['note']}`",
        ]
    )

    out_md = output_json.with_name(f"{output_json.stem}.md")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_md


def select_codec_inputs(payload_paths: List[Path]) -> List[Path]:
    files = [p for p in payload_paths if p.exists() and p.is_file()]
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files[:10]


def run_center(profile: str, include_rsa: bool, rsa_folds: int) -> Tuple[Path, Path]:
    ts = int(time.time())
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    mc_samples = 12000 if profile == "quick" else 30000

    stages: Dict[str, Dict[str, object]] = {}
    stages["das"] = run_das_validation(monte_carlo_samples=mc_samples)
    stages["dual"] = run_dual_conjugate_tuning()

    if include_rsa:
        stages["rsa"] = run_rsa_parallel_cv(k_folds=rsa_folds)

    # Encode stage outputs to files first, then benchmark codec on those artifacts.
    stage_files: List[Path] = []
    for name, stage in stages.items():
        p = out_dir / f"joint_center_{name}_{ts}.json"
        p.write_text(json.dumps(stage["report"], ensure_ascii=False, indent=2), encoding="utf-8")
        stage_files.append(p)

    codec_inputs = select_codec_inputs(stage_files)
    codec_out = out_dir / f"joint_center_codec_roundtrip_{ts}"
    stages["codec"] = run_codec_regression(codec_inputs, codec_out)

    aggregate = aggregate_verdict(stages)

    payload = {
        "meta": {
            "timestamp": ts,
            "profile": profile,
            "include_rsa": include_rsa,
            "rsa_folds": rsa_folds,
            "artifact_hashes": {str(p): sha256_file(p) for p in stage_files},
        },
        "stages": stages,
        "aggregate": aggregate,
        "codec_inputs": [str(p) for p in codec_inputs],
        "codec_output_dir": str(codec_out),
    }

    out_json = out_dir / f"trusted_joint_agi_quantum_center_{ts}.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md = write_report_markdown(out_json, payload)
    return out_json, out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Trusted joint AGI-quantized system center orchestrator")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--skip-rsa", action="store_true", help="Skip RSA parallel CV stage")
    parser.add_argument("--rsa-folds", type=int, default=2)
    args = parser.parse_args()

    out_json, out_md = run_center(
        profile=args.profile,
        include_rsa=not args.skip_rsa,
        rsa_folds=max(1, args.rsa_folds),
    )
    print("Trusted joint AGI-quantized center completed")
    print(f"Data: {out_json}")
    print(f"Report: {out_md}")


if __name__ == "__main__":
    main()
