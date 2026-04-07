#!/usr/bin/env python3
"""Slowly read arXiv papers and synthesize local capability merge/upgrade plans.

This script does three things in one pipeline:
1. Slowly pull papers from multiple arXiv disciplines with rate limiting.
2. Build a local capability profile from the repository architecture.
3. Generate merge/upgrade actions based on demand-vs-strength gap analysis.

Outputs (timestamped run directory):
- arxiv_papers.json
- local_capability_profile.json
- capability_merge_upgrade_plan.json
- capability_merge_upgrade_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CATEGORIES: List[str] = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "cs.RO",
    "cs.CV",
    "stat.ML",
    "eess.AS",
    "eess.IV",
    "math.OC",
    "math.PR",
    "quant-ph",
    "physics.comp-ph",
    "q-bio.NC",
    "econ.EM",
]


CAPABILITY_KEYWORDS: Dict[str, List[str]] = {
    "quantum_computation": [
        "quantum",
        "qubit",
        "entanglement",
        "quantization",
        "hamiltonian",
        "quant-ph",
        "qft",
    ],
    "agent_autonomy": [
        "agent",
        "autonomous",
        "self evolution",
        "self-improvement",
        "planner",
        "tool use",
        "workflow",
        "orchestr",
    ],
    "reasoning_planning": [
        "reasoning",
        "logic",
        "planning",
        "search",
        "inference",
        "theorem",
        "chain of thought",
    ],
    "multimodal_perception": [
        "vision",
        "image",
        "video",
        "audio",
        "speech",
        "multimodal",
        "cross-modal",
    ],
    "knowledge_memory": [
        "knowledge",
        "memory",
        "retrieval",
        "index",
        "embedding",
        "dataset",
        "corpus",
    ],
    "optimization_efficiency": [
        "optimization",
        "efficient",
        "distillation",
        "compression",
        "sparse",
        "accelerat",
        "low-rank",
    ],
    "robustness_safety": [
        "robust",
        "safety",
        "secure",
        "alignment",
        "fault",
        "monitor",
        "recovery",
        "verification",
    ],
    "math_foundations": [
        "manifold",
        "topology",
        "geometry",
        "algebra",
        "spectral",
        "operator",
        "differential",
    ],
    "systems_infrastructure": [
        "server",
        "api",
        "docker",
        "distributed",
        "scheduler",
        "latency",
        "throughput",
    ],
}


IGNORE_DIR_TOKENS: Tuple[str, ...] = (
    "/.git/",
    "/.venv/",
    "/venv/",
    "/__pycache__/",
    "/node_modules/",
    "/external/",
)


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class PaperEntry:
    category: str
    paper_id: str
    title: str
    summary: str
    published: str
    updated: str
    authors: List[str]
    terms: List[str]
    tags: List[str]


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _count_keyword(text: str, keyword: str) -> int:
    normalized = keyword.strip().lower()
    if not normalized:
        return 0
    if " " in normalized or "-" in normalized:
        return text.count(normalized)
    pattern = rf"\b{re.escape(normalized)}"
    return len(re.findall(pattern, text))


def score_capabilities(text: str) -> Dict[str, int]:
    body = text.lower()
    scores: Dict[str, int] = {}
    for capability, keywords in CAPABILITY_KEYWORDS.items():
        total = 0
        for keyword in keywords:
            total += _count_keyword(body, keyword)
        scores[capability] = total
    return scores


def parse_arxiv_entry(category: str, node: ET.Element) -> PaperEntry:
    title = _clean_text(node.findtext("atom:title", default="", namespaces=ATOM_NS))
    summary = _clean_text(node.findtext("atom:summary", default="", namespaces=ATOM_NS))
    published = _clean_text(node.findtext("atom:published", default="", namespaces=ATOM_NS))
    updated = _clean_text(node.findtext("atom:updated", default="", namespaces=ATOM_NS))
    paper_id = _clean_text(node.findtext("atom:id", default="", namespaces=ATOM_NS))

    authors: List[str] = []
    for author in node.findall("atom:author", namespaces=ATOM_NS):
        name = _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
        if name:
            authors.append(name)

    terms: List[str] = []
    for cat in node.findall("atom:category", namespaces=ATOM_NS):
        term = _clean_text(cat.attrib.get("term", ""))
        if term:
            terms.append(term)

    capability_scores = score_capabilities(" ".join([title, summary, " ".join(terms)]))
    tags = [
        key
        for key, score in sorted(capability_scores.items(), key=lambda kv: kv[1], reverse=True)
        if score > 0
    ][:4]

    return PaperEntry(
        category=category,
        paper_id=paper_id,
        title=title,
        summary=summary,
        published=published,
        updated=updated,
        authors=authors,
        terms=terms,
        tags=tags,
    )


def _parse_arxiv_time(value: str) -> datetime | None:
    raw = value.strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue
    return None


def fetch_arxiv_category(
    category: str,
    *,
    max_results: int,
    timeout_sec: float,
) -> List[PaperEntry]:
    query = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )

    endpoints = [
        f"https://export.arxiv.org/api/query?{query}",
        f"http://export.arxiv.org/api/query?{query}",
    ]

    last_err: Exception | None = None
    for endpoint in endpoints:
        req = urllib.request.Request(
            endpoint,
            headers={
                "User-Agent": "H2Q-Evo-ArXivUpgrade/1.0 (+https://arxiv.org/)"
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                body = resp.read()
            root = ET.fromstring(body)
            entries: List[PaperEntry] = []
            for node in root.findall("atom:entry", namespaces=ATOM_NS):
                entries.append(parse_arxiv_entry(category, node))
            return entries
        except Exception as exc:
            last_err = exc

    if last_err is not None:
        raise last_err
    return []


def _is_ignored(path: Path) -> bool:
    normalized = "/" + str(path).replace("\\", "/") + "/"
    return any(token in normalized for token in IGNORE_DIR_TOKENS)


def load_symbol_index(analysis_root: Path) -> Dict[str, str]:
    try:
        from project_graph import generate_interface_map

        _report, index = generate_interface_map(str(analysis_root))
        if isinstance(index, dict):
            return {str(k): str(v) for k, v in index.items()}
    except Exception:
        pass
    return {}


def analyze_local_capabilities(
    analysis_root: Path,
) -> Dict[str, Any]:
    py_files: List[Path] = []
    for file_path in analysis_root.rglob("*.py"):
        if _is_ignored(file_path):
            continue
        py_files.append(file_path)

    capability_hits = {name: 0 for name in CAPABILITY_KEYWORDS}
    capability_file_counts = {name: 0 for name in CAPABILITY_KEYWORDS}
    capability_file_scores: Dict[str, List[Tuple[str, int]]] = {name: [] for name in CAPABILITY_KEYWORDS}

    for file_path in py_files:
        try:
            raw = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Use path+head chunk for robust and bounded analysis cost.
        sample = f"{file_path} {raw[:24000]}"
        scores = score_capabilities(sample)

        for capability, score in scores.items():
            if score <= 0:
                continue
            capability_hits[capability] += score
            capability_file_counts[capability] += 1
            rel_path = str(file_path.relative_to(PROJECT_ROOT))
            capability_file_scores[capability].append((rel_path, score))

    total_files = max(1, len(py_files))
    capability_profile: Dict[str, Any] = {}
    for capability in CAPABILITY_KEYWORDS:
        hits = capability_hits[capability]
        files = capability_file_counts[capability]
        file_coverage = files / float(total_files)
        # Log-scale normalize for stable score under large repositories.
        strength = min(1.0, math.log1p(hits) / math.log1p(1200.0))

        top_files = sorted(
            capability_file_scores[capability],
            key=lambda item: item[1],
            reverse=True,
        )[:8]

        capability_profile[capability] = {
            "keyword_hits": hits,
            "covered_files": files,
            "file_coverage": round(file_coverage, 6),
            "local_strength": round(strength, 6),
            "top_files": [
                {"path": path, "score": score}
                for path, score in top_files
            ],
        }

    symbol_index = load_symbol_index(analysis_root)
    return {
        "analysis_root": str(analysis_root),
        "python_file_count": len(py_files),
        "capabilities": capability_profile,
        "symbol_index_size": len(symbol_index),
    }


def build_upgrade_plan(
    papers: Sequence[PaperEntry],
    local_profile: Dict[str, Any],
) -> Dict[str, Any]:
    total_papers = max(1, len(papers))
    demand_counts = {name: 0 for name in CAPABILITY_KEYWORDS}

    for paper in papers:
        for tag in paper.tags:
            if tag in demand_counts:
                demand_counts[tag] += 1

    local_caps = local_profile.get("capabilities", {})

    capability_rows: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []

    for capability in CAPABILITY_KEYWORDS:
        demand_ratio = demand_counts[capability] / float(total_papers)
        local_strength = float(local_caps.get(capability, {}).get("local_strength", 0.0))
        gap = demand_ratio - local_strength

        if demand_ratio >= 0.08 and gap > 0.20:
            action_type = "upgrade"
            priority = "high"
        elif demand_ratio >= 0.05 and gap > 0.10:
            action_type = "upgrade"
            priority = "medium"
        elif demand_ratio >= 0.05 and local_strength >= 0.40:
            action_type = "merge"
            priority = "medium"
        elif demand_ratio >= 0.03:
            action_type = "merge"
            priority = "low"
        else:
            action_type = "monitor"
            priority = "low"

        top_files = [
            item["path"]
            for item in local_caps.get(capability, {}).get("top_files", [])
        ][:5]

        if action_type == "upgrade":
            suggested_tasks = [
                f"Design capability extension interfaces for {capability}",
                f"Add benchmark/evaluation hooks for {capability}",
                f"Integrate {capability} pipeline into long-run evolution orchestration",
            ]
        elif action_type == "merge":
            suggested_tasks = [
                f"Unify duplicated {capability} implementations across modules",
                f"Promote strongest {capability} components into shared APIs",
            ]
        else:
            suggested_tasks = [
                f"Track arXiv demand trend for {capability}",
            ]

        rationale = (
            f"demand_ratio={demand_ratio:.3f}, local_strength={local_strength:.3f}, "
            f"gap={gap:.3f}"
        )

        row = {
            "capability": capability,
            "demand_count": demand_counts[capability],
            "demand_ratio": round(demand_ratio, 6),
            "local_strength": round(local_strength, 6),
            "gap": round(gap, 6),
            "recommended_action": action_type,
            "priority": priority,
            "candidate_local_modules": top_files,
            "rationale": rationale,
        }
        capability_rows.append(row)

        actions.append(
            {
                "capability": capability,
                "action": action_type,
                "priority": priority,
                "rationale": rationale,
                "candidate_local_modules": top_files,
                "suggested_tasks": suggested_tasks,
            }
        )

    capability_rows.sort(key=lambda item: (item["priority"], item["gap"]), reverse=True)
    actions.sort(key=lambda item: (item["priority"], item["rationale"]), reverse=True)

    return {
        "paper_count": len(papers),
        "capability_demand": demand_counts,
        "capability_gap_rows": capability_rows,
        "actions": actions,
    }


def write_report_markdown(
    run_dir: Path,
    *,
    metadata: Dict[str, Any],
    papers: Sequence[PaperEntry],
    local_profile: Dict[str, Any],
    upgrade_plan: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# arXiv Capability Merge and Upgrade Report")
    lines.append("")
    lines.append(f"- generated_at: {metadata['generated_at']}")
    lines.append(f"- run_dir: {run_dir}")
    lines.append(f"- categories: {', '.join(metadata['categories'])}")
    lines.append(f"- paper_count: {len(papers)}")
    lines.append(f"- local_python_files: {local_profile.get('python_file_count', 0)}")
    lines.append("")

    lines.append("## Top Upgrade or Merge Priorities")
    lines.append("")
    for action in upgrade_plan.get("actions", [])[:12]:
        lines.append(
            f"- [{action['priority']}] {action['capability']} -> {action['action']} "
            f"({action['rationale']})"
        )
        modules = action.get("candidate_local_modules", [])
        if modules:
            lines.append(f"  modules: {', '.join(modules[:3])}")
        tasks = action.get("suggested_tasks", [])
        if tasks:
            lines.append(f"  task: {tasks[0]}")

    lines.append("")
    lines.append("## Sample Papers")
    lines.append("")
    for paper in papers[:20]:
        lines.append(
            f"- {paper.title} | category={paper.category} | tags={','.join(paper.tags) if paper.tags else 'none'}"
        )

    (run_dir / "capability_merge_upgrade_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slowly read arXiv and generate local capability merge/upgrade plan"
    )
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Comma-separated arXiv categories",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=6,
        help="Papers to fetch per category",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=72,
        help="Optional cap over total paper count",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=4.0,
        help="Delay between category requests for slow crawling",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=25.0,
        help="HTTP timeout seconds",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Keep papers published in last N days (0 disables filtering)",
    )
    parser.add_argument(
        "--analysis-root",
        default="h2q_project",
        help="Repository subdirectory for local capability analysis",
    )
    parser.add_argument(
        "--output-dir",
        default="h2q_project/reports/arxiv_capability_upgrade",
        help="Output base directory",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run directory name",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    if not categories:
        print("No categories configured", file=sys.stderr)
        return 2

    output_base = Path(args.output_dir)
    if not output_base.is_absolute():
        output_base = (PROJECT_ROOT / output_base).resolve()
    else:
        output_base = output_base.resolve()

    run_name = args.run_name.strip() or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = (output_base / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("arXiv slow capability upgrade pipeline started")
    print(f"run_dir: {run_dir}")
    print(f"categories: {', '.join(categories)}")
    print("=" * 72)

    all_papers: List[PaperEntry] = []
    fetch_errors: List[Dict[str, str]] = []

    min_time: datetime | None = None
    if args.days > 0:
        min_time = datetime.now(timezone.utc) - timedelta(days=max(1, args.days))

    for idx, category in enumerate(categories):
        try:
            papers = fetch_arxiv_category(
                category,
                max_results=max(1, args.per_category),
                timeout_sec=max(5.0, args.timeout_seconds),
            )
        except Exception as exc:
            fetch_errors.append({"category": category, "error": str(exc)})
            papers = []

        if min_time is not None and papers:
            filtered: List[PaperEntry] = []
            for paper in papers:
                published = _parse_arxiv_time(paper.published)
                if published is None or published >= min_time:
                    filtered.append(paper)
            papers = filtered

        all_papers.extend(papers)
        print(f"[arxiv] {category}: {len(papers)} papers")

        if args.max_total > 0 and len(all_papers) >= args.max_total:
            all_papers = all_papers[: args.max_total]
            break

        if idx < len(categories) - 1:
            time.sleep(max(0.0, args.delay_seconds))

    analysis_root = Path(args.analysis_root)
    if not analysis_root.is_absolute():
        analysis_root = (PROJECT_ROOT / analysis_root).resolve()
    else:
        analysis_root = analysis_root.resolve()

    local_profile = analyze_local_capabilities(analysis_root)
    upgrade_plan = build_upgrade_plan(all_papers, local_profile)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "categories": categories,
        "per_category": args.per_category,
        "max_total": args.max_total,
        "delay_seconds": args.delay_seconds,
        "days": args.days,
        "analysis_root": str(analysis_root),
        "fetch_errors": fetch_errors,
    }

    (run_dir / "arxiv_papers.json").write_text(
        json.dumps([asdict(paper) for paper in all_papers], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "local_capability_profile.json").write_text(
        json.dumps(local_profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "capability_merge_upgrade_plan.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "upgrade_plan": upgrade_plan,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report_markdown(
        run_dir,
        metadata=metadata,
        papers=all_papers,
        local_profile=local_profile,
        upgrade_plan=upgrade_plan,
    )

    print(json.dumps({"run_dir": str(run_dir), "papers": len(all_papers)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
