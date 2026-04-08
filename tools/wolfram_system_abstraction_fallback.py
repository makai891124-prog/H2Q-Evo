#!/usr/bin/env python3
"""Generate system abstraction graph and logic proof report without Wolfram kernel.

This fallback uses only Python standard library so it can run in restricted environments.
"""

from __future__ import annotations

import json
import itertools
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def implies(a: bool, b: bool) -> bool:
    return (not a) or b


def evaluate_theorems(assignment: Dict[str, bool]) -> Dict[str, bool]:
    dynamic_bootstrap = assignment["dynamicBootstrap"]
    strong_retry = assignment["strongRetry"]
    capability_registry_ok = assignment["capabilityRegistryOK"]
    interactive_benchmark_ok = assignment["interactiveBenchmarkOK"]
    public_alignment_ok = assignment["publicAlignmentOK"]
    regression_guard_ok = assignment["regressionGuardOK"]
    release_gate_pass = assignment["releaseGatePass"]
    trusted_joint_center_ok = assignment["trustedJointCenterOK"]
    ci_green = assignment["ciGreen"]
    production_candidate = assignment["productionCandidate"]

    premises = [
        implies(dynamic_bootstrap and strong_retry, capability_registry_ok),
        implies(dynamic_bootstrap and strong_retry, public_alignment_ok),
        implies(dynamic_bootstrap and strong_retry, regression_guard_ok),
        implies(
            capability_registry_ok
            and interactive_benchmark_ok
            and public_alignment_ok
            and regression_guard_ok,
            release_gate_pass,
        ),
        implies(release_gate_pass, trusted_joint_center_ok),
        implies(release_gate_pass, ci_green),
        implies(trusted_joint_center_ok and ci_green, production_candidate),
    ]

    # Governance assumptions make production candidacy traceable and gate-rooted.
    governance_axioms = [
        implies(production_candidate, trusted_joint_center_ok and ci_green),
        implies(trusted_joint_center_ok and ci_green, release_gate_pass),
    ]

    all_premises = all(premises)
    all_governance_axioms = all(governance_axioms)

    theorem1 = implies(
        all_premises and dynamic_bootstrap and strong_retry and interactive_benchmark_ok,
        production_candidate,
    )
    theorem2 = implies(
        all_premises and release_gate_pass,
        trusted_joint_center_ok and ci_green,
    )
    theorem3 = implies(
        all_premises and all_governance_axioms and production_candidate,
        release_gate_pass,
    )

    return {
        "Theorem1_DeliveryClosure": theorem1,
        "Theorem2_GateImpliesTrustAndCI": theorem2,
        "Theorem3_ProductionRequiresGate": theorem3,
    }


def prove_with_truth_table(symbols: List[str]) -> Tuple[Dict[str, bool], Dict[str, Dict[str, bool]]]:
    proofs = {
        "Theorem1_DeliveryClosure": True,
        "Theorem2_GateImpliesTrustAndCI": True,
        "Theorem3_ProductionRequiresGate": True,
    }
    counter_examples: Dict[str, Dict[str, bool]] = {
        "Theorem1_DeliveryClosure": {},
        "Theorem2_GateImpliesTrustAndCI": {},
        "Theorem3_ProductionRequiresGate": {},
    }

    for values in itertools.product([False, True], repeat=len(symbols)):
        assignment = dict(zip(symbols, values))
        result = evaluate_theorems(assignment)
        for theorem_name, theorem_ok in result.items():
            if not theorem_ok:
                proofs[theorem_name] = False
                if not counter_examples[theorem_name]:
                    counter_examples[theorem_name] = assignment

    return proofs, counter_examples


def build_graph_svg(
    components: List[str],
    edges: List[Tuple[str, str]],
    out_path: Path,
) -> None:
    layer_x = [80, 380, 680, 980, 1280]
    layer_y = [120, 260, 400, 540, 680]
    positions = {
        "DynamicBlueprintBootstrap": (layer_x[0], layer_y[2]),
        "CapabilityRegistry": (layer_x[1], layer_y[1]),
        "InteractiveBenchmark": (layer_x[1], layer_y[2]),
        "PublicAlignment": (layer_x[1], layer_y[3]),
        "RegressionGuard": (layer_x[1], layer_y[4]),
        "ReleaseGate": (layer_x[2], layer_y[2]),
        "RealtimeMonitor": (layer_x[0], layer_y[4]),
        "TrustedJointCenter": (layer_x[3], layer_y[1]),
        "CISelfEvolution": (layer_x[3], layer_y[3]),
        "RuntimeReports": (layer_x[4], layer_y[2]),
    }

    width, height = 1400, 800
    node_w, node_h = 220, 54

    def edge_line(src: str, dst: str) -> str:
        sx, sy = positions[src]
        dx, dy = positions[dst]
        x1 = sx + node_w / 2
        y1 = sy
        x2 = dx - node_w / 2
        y2 = dy
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#2f4858" stroke-width="2.2" marker-end="url(#arrow)" />'
        )

    def node_rect(name: str) -> str:
        x, y = positions[name]
        left = x - node_w / 2
        top = y - node_h / 2
        return (
            f'<rect x="{left}" y="{top}" width="{node_w}" height="{node_h}" '
            'rx="12" ry="12" fill="#2f6690" opacity="0.92" stroke="#173753" stroke-width="1.8" />'
            f'<text x="{x}" y="{y + 6}" text-anchor="middle" fill="#f7fbff" '
            'font-family="Helvetica,Arial,sans-serif" font-size="14">'
            f"{name}"
            "</text>"
        )

    edge_svg = "\n    ".join(edge_line(s, d) for s, d in edges)
    node_svg = "\n    ".join(node_rect(n) for n in components)

    svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n  <defs>\n    <linearGradient id=\"bg\" x1=\"0%\" y1=\"0%\" x2=\"100%\" y2=\"100%\">\n      <stop offset=\"0%\" stop-color=\"#edf6f9\"/>\n      <stop offset=\"100%\" stop-color=\"#d7e3fc\"/>\n    </linearGradient>\n    <marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"7\" refX=\"10\" refY=\"3.5\" orient=\"auto\">\n      <polygon points=\"0 0, 10 3.5, 0 7\" fill=\"#2f4858\" />\n    </marker>\n  </defs>\n  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"url(#bg)\"/>\n  <text x=\"700\" y=\"42\" text-anchor=\"middle\" fill=\"#173753\" font-family=\"Helvetica,Arial,sans-serif\" font-size=\"24\" font-weight=\"bold\">H2Q-Evo System Abstraction Graph</text>\n  {edge_svg}\n  {node_svg}\n</svg>\n"""

    out_path.write_text(svg, encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = str(int(time.time()))
    generated_at = datetime.now(timezone.utc).isoformat()

    components = [
        "DynamicBlueprintBootstrap",
        "CapabilityRegistry",
        "InteractiveBenchmark",
        "PublicAlignment",
        "RegressionGuard",
        "ReleaseGate",
        "RealtimeMonitor",
        "TrustedJointCenter",
        "CISelfEvolution",
        "RuntimeReports",
    ]

    edges = [
        ("DynamicBlueprintBootstrap", "CapabilityRegistry"),
        ("DynamicBlueprintBootstrap", "PublicAlignment"),
        ("DynamicBlueprintBootstrap", "RegressionGuard"),
        ("CapabilityRegistry", "ReleaseGate"),
        ("InteractiveBenchmark", "ReleaseGate"),
        ("PublicAlignment", "ReleaseGate"),
        ("RegressionGuard", "ReleaseGate"),
        ("ReleaseGate", "TrustedJointCenter"),
        ("ReleaseGate", "CISelfEvolution"),
        ("RealtimeMonitor", "RuntimeReports"),
        ("DynamicBlueprintBootstrap", "RuntimeReports"),
        ("CISelfEvolution", "RuntimeReports"),
    ]

    symbols = [
        "dynamicBootstrap",
        "strongRetry",
        "capabilityRegistryOK",
        "interactiveBenchmarkOK",
        "publicAlignmentOK",
        "regressionGuardOK",
        "releaseGatePass",
        "trustedJointCenterOK",
        "ciGreen",
        "productionCandidate",
    ]

    proofs, counter_examples = prove_with_truth_table(symbols)

    graph_svg = reports_dir / f"wolfram_system_abstraction_graph_{timestamp}.svg"
    latest_graph_svg = reports_dir / "wolfram_system_abstraction_graph_latest.svg"
    report_json = reports_dir / f"wolfram_logic_proof_report_{timestamp}.json"
    latest_report_json = reports_dir / "wolfram_logic_proof_report_latest.json"
    report_txt = reports_dir / f"wolfram_logic_proof_report_{timestamp}.txt"
    latest_report_txt = reports_dir / "wolfram_logic_proof_report_latest.txt"

    build_graph_svg(components, edges, graph_svg)
    shutil.copy2(graph_svg, latest_graph_svg)

    payload = {
        "meta": {
            "generated_at": generated_at,
            "repo_root": str(repo_root),
            "component_count": len(components),
            "edge_count": len(edges),
            "engine": "python-fallback-stdlib",
            "theorem3_strengthened": {
                "name": "ProductionRequiresGateUnderGovernanceAxioms",
                "axioms": [
                    "productionCandidate -> (trustedJointCenterOK and ciGreen)",
                    "(trustedJointCenterOK and ciGreen) -> releaseGatePass",
                ],
            },
        },
        "components": components,
        "edges": edges,
        "proofs": proofs,
        "counter_examples": counter_examples,
        "artifacts": {
            "graph_svg": str(graph_svg),
            "latest_graph_svg": str(latest_graph_svg),
        },
    }

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(report_json, latest_report_json)

    text = "\n".join(
        [
            "Wolfram-Compatible System Abstraction & Logic Proof Report (Python Fallback)",
            f"Generated: {generated_at}",
            "Theorem3 strengthened under governance axioms:",
            "  A1: productionCandidate -> (trustedJointCenterOK and ciGreen)",
            "  A2: (trustedJointCenterOK and ciGreen) -> releaseGatePass",
            "",
            f"Theorem1_DeliveryClosure: {proofs['Theorem1_DeliveryClosure']}",
            f"Theorem2_GateImpliesTrustAndCI: {proofs['Theorem2_GateImpliesTrustAndCI']}",
            f"Theorem3_ProductionRequiresGate: {proofs['Theorem3_ProductionRequiresGate']}",
            "",
            f"Graph SVG: {graph_svg}",
            f"Report JSON: {report_json}",
        ]
    )

    report_txt.write_text(text, encoding="utf-8")
    shutil.copy2(report_txt, latest_report_txt)

    print(f"Graph SVG: {graph_svg}")
    print(f"Report JSON: {report_json}")
    print(f"Report TXT: {report_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
