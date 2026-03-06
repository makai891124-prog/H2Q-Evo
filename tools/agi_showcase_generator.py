import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from das_experimental_validator import DASExperimentalValidator
from tools.dual_conjugate_joint_tuning import evaluate_dual_conjugate_modes
from tools.industrial_realtime_codec import batch_convert


def latest_center_json() -> Path:
    files = sorted(Path("reports").glob("trusted_joint_agi_quantum_center_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError("No trusted_joint_agi_quantum_center_*.json found. Run center orchestrator first.")
    return files[-1]


def load_center(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_live_checks() -> Dict[str, object]:
    # A lighter but real validation pass to demonstrate online functionality.
    das = DASExperimentalValidator(monte_carlo_samples=4000, seed=42, precision_eps=1e-18).build_statistical_report()
    dual = evaluate_dual_conjugate_modes()

    tmp_in = Path("reports")
    tmp_out = Path("reports") / f"agi_showcase_codec_batch_{int(time.time())}"
    codec = batch_convert(
        input_dir=tmp_in,
        output_dir=tmp_out,
        mode="compress",
        workers=4,
        recursive=False,
        pattern="*.md",
        level=6,
    )

    return {
        "das_live": {
            "decision_grade_ready": bool(das["verdict"].get("decision_grade_ready", False)),
            "confidence": float(das["confidence"].get("isomorphic_confidence_score", 0.0)),
            "dual_conjugate_aligned_pass": bool(das["verdict"].get("dual_conjugate_aligned_pass", False)),
        },
        "dual_live": {
            "best_mode": dual["summary"].get("best_mode"),
            "aligned_corr": float(dual["summary"].get("aligned_corr", 0.0)),
            "aligned_mae": float(dual["summary"].get("aligned_mae", 1e9)),
            "sample_count": int(dual.get("sample_count", 0)),
        },
        "codec_live": {
            "task_count": int(codec.get("task_count", 0)),
            "ok_count": int(codec.get("ok_count", 0)),
            "failed_count": int(codec.get("failed_count", 0)),
            "elapsed_seconds": float(codec.get("elapsed_seconds", 0.0)),
            "output_dir": str(tmp_out),
        },
    }


def compute_features(center: Dict[str, object], live: Dict[str, object]) -> Dict[str, float]:
    stages = center.get("stages", {})
    agg = center.get("aggregate", {})

    das_conf = float(stages.get("das", {}).get("key", {}).get("isomorphic_confidence_score", 0.0))
    dual_corr = float(stages.get("dual", {}).get("key", {}).get("aligned_corr", 0.0))
    codec_ratio = float(stages.get("codec", {}).get("key", {}).get("mean_ratio", 0.0))
    rsa_spd = float(stages.get("rsa", {}).get("key", {}).get("mean_parallel_speedup", 0.0))
    trust = float(agg.get("trust_score", 0.0))

    # Normalize to [0,1] for visual capability map.
    feat = {
        "物理一致性": min(1.0, max(0.0, das_conf)),
        "双复数同构": min(1.0, max(0.0, dual_corr)),
        "工程完整性": min(1.0, max(0.0, codec_ratio / 5.0)),
        "并行效率": min(1.0, max(0.0, rsa_spd / 1.2)) if rsa_spd > 0 else 0.0,
        "可信聚合": min(1.0, max(0.0, trust)),
    }

    # Blend with live checks so showcase is not only historical snapshot.
    live_conf = float(live["das_live"]["confidence"])
    feat["物理一致性"] = float((feat["物理一致性"] + min(1.0, max(0.0, live_conf))) / 2.0)
    return feat


def render_dashboard(features: Dict[str, float], out_png: Path) -> None:
    labels = {
        "物理一致性": "Physics",
        "双复数同构": "Dual-Conjugate",
        "工程完整性": "Engineering",
        "并行效率": "Parallel",
        "可信聚合": "Trust",
    }
    names = [labels.get(k, k) for k in features.keys()]
    vals = [features[k] for k in features.keys()]

    plt.figure(figsize=(10, 6))
    y = np.arange(len(names))
    plt.barh(y, vals, color=["#2364AA", "#3DA35D", "#F49D37", "#D7263D", "#6C5CE7"])
    plt.xlim(0, 1)
    plt.yticks(y, names)
    plt.xlabel("Normalized score (0-1)")
    plt.title("Joint AGI-Quantum System Real Capability Showcase")
    for i, v in enumerate(vals):
        plt.text(min(v + 0.02, 0.97), i, f"{v:.3f}", va="center", fontsize=10)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def render_icon_svg(features: Dict[str, float], out_svg: Path) -> None:
    trust = features.get("可信聚合", 0.0)
    color = "#1E8449" if trust >= 0.75 else "#B9770E" if trust >= 0.6 else "#C0392B"
    score = int(round(trust * 100))

    w, h = 512, 512
    cx, cy = 256, 256
    r = 170

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="#0B132B"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#1C2541" stroke="{color}" stroke-width="10"/>',
        f'<circle cx="{cx}" cy="{cy}" r="{int(r*0.66)}" fill="#3A506B" opacity="0.35"/>',
        f'<text x="{cx}" y="210" fill="#EAF4F4" font-size="34" text-anchor="middle" font-family="Arial">AGI-Q Center</text>',
        f'<text x="{cx}" y="270" fill="{color}" font-size="72" text-anchor="middle" font-weight="bold" font-family="Arial">{score}</text>',
        f'<text x="{cx}" y="315" fill="#EAF4F4" font-size="24" text-anchor="middle" font-family="Arial">Trust Score</text>',
        f'<text x="{cx}" y="455" fill="#BFD7EA" font-size="18" text-anchor="middle" font-family="Arial">H2Q-Evo Real Module Showcase</text>',
        '</svg>',
    ]
    out_svg.write_text("".join(svg), encoding="utf-8")


def write_report(center_path: Path, center: Dict[str, object], live: Dict[str, object], features: Dict[str, float], out_md: Path, icon_svg: Path, dashboard_png: Path) -> None:
    agg = center.get("aggregate", {})
    stages = center.get("stages", {})
    das = stages.get("das", {}).get("key", {})
    dual = stages.get("dual", {}).get("key", {})
    rsa = stages.get("rsa", {}).get("key", {})
    codec = stages.get("codec", {}).get("key", {})

    lines = [
        "# 联合AGI量子化系统真实功能展示最终中文报告",
        "",
        "## 1. 展示目标",
        "",
        "本报告展示的是基于仓库真实实现代码的可执行能力，不是静态宣称。",
        "展示结果来自实际调用中心编排器、DAS主验证链、双复数联调与工业编解码模块。",
        "",
        "## 2. 真实调用链",
        "",
        f"- 中心基线数据：`{center_path}`",
        "- 调用模块：`das_experimental_validator.py`、`tools/dual_conjugate_joint_tuning.py`、`tools/rsa_parallel_cv_no_timefold.py`、`tools/industrial_realtime_codec.py`",
        "- 现场复核：脚本内再次真实运行小规模 DAS + Dual + Codec 批处理。",
        "",
        "## 3. 核心功能特性（真实指标）",
        "",
        f"- DAS 决策级可用：`{das.get('decision_grade_ready', False)}`",
        f"- DAS 可信分：`{das.get('isomorphic_confidence_score', 0.0):.4f}`",
        f"- 双复数最优口径：`{dual.get('best_mode', 'unknown')}`",
        f"- 双复数 aligned corr/mae：`{dual.get('aligned_corr', 0.0):.4f}` / `{dual.get('aligned_mae', 0.0):.4f}`",
        f"- RSA 并行加速均值：`{rsa.get('mean_parallel_speedup', 0.0):.3f}x`",
        f"- 工业编解码完整性：`{codec.get('all_checksum_match', False)}`",
        f"- 工业编解码平均压缩比：`{codec.get('mean_ratio', 0.0):.3f}x`",
        "",
        "## 4. 现场复核结果",
        "",
        f"- live DAS decision/confidence：`{live['das_live']['decision_grade_ready']}` / `{live['das_live']['confidence']:.4f}`",
        f"- live Dual mode/corr/mae：`{live['dual_live']['best_mode']}` / `{live['dual_live']['aligned_corr']:.4f}` / `{live['dual_live']['aligned_mae']:.4f}`",
        f"- live Codec batch ok/task：`{live['codec_live']['ok_count']}/{live['codec_live']['task_count']}`，失败：`{live['codec_live']['failed_count']}`",
        f"- live Codec 输出目录：`{live['codec_live']['output_dir']}`",
        "",
        "## 5. 能力归一化画像",
        "",
        *(f"- {k}：`{v:.3f}`" for k, v in features.items()),
        "",
        "## 6. 综合判定",
        "",
        f"- 聚合 gates：`{agg.get('gates', {})}`",
        f"- Trust score：`{agg.get('trust_score', 0.0):.4f}`",
        f"- Trusted ready：`{agg.get('trusted_ready', False)}`",
        "- 结论：系统具备“可复验的联合编排能力与物理-工程双链路验证能力”；是否达到更高层通用AGI需额外任务域与长期自治证据。",
        "",
        "## 7. 展示附件",
        "",
        f"- 展示图标（SVG）：`{icon_svg}`",
        f"- 功能特性图（PNG）：`{dashboard_png}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AGI feature showcase icon and Chinese detailed report")
    parser.add_argument("--center-json", default="", help="Path to trusted_joint_agi_quantum_center_*.json")
    args = parser.parse_args()

    center_path = Path(args.center_json) if args.center_json else latest_center_json()
    center = load_center(center_path)
    live = run_live_checks()
    features = compute_features(center, live)

    ts = int(time.time())
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    icon_svg = out_dir / f"agi_showcase_icon_{ts}.svg"
    dashboard_png = out_dir / f"agi_showcase_dashboard_{ts}.png"
    out_json = out_dir / f"agi_showcase_{ts}.json"
    out_md = out_dir / f"AGI真实功能展示最终报告_{ts}.md"

    render_icon_svg(features, icon_svg)
    render_dashboard(features, dashboard_png)

    payload = {
        "meta": {
            "timestamp": ts,
            "center_json": str(center_path),
        },
        "features": features,
        "live_checks": live,
        "center_aggregate": center.get("aggregate", {}),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(center_path, center, live, features, out_md, icon_svg, dashboard_png)

    print("AGI showcase generated")
    print(f"Icon: {icon_svg}")
    print(f"Dashboard: {dashboard_png}")
    print(f"Data: {out_json}")
    print(f"Report: {out_md}")


if __name__ == "__main__":
    main()
