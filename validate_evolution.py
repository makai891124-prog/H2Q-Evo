"""
Validation script for self-programming capability (Live Test).

Generates random CSV, triggers evolution_system.process, and verifies:
- new tool file appears in custom_tools/
- tool uses pandas or csv
- agent result matches locally computed truth

Outputs evolution_proof.json
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import statistics

import evolution_system

ROOT = Path(__file__).resolve().parent
H2Q_PROJECT = ROOT / "h2q_project"
TOOLS_DIR = H2Q_PROJECT / "custom_tools"


def create_random_csv(path: Path, rows: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Value"])
        writer.writeheader()
        for _ in range(rows):
            writer.writerow({"Value": f"{random.random():.10f}"})


def compute_truth(path: Path, hour: int) -> float:
    values = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values.append(float(row["Value"]))
    std = statistics.pstdev(values) if values else 0.0
    return std * hour


def get_new_tool_file(before: set[str], after: set[str]) -> Optional[Path]:
    new_files = sorted(list(after - before))
    if not new_files:
        return None
    return TOOLS_DIR / new_files[-1]


def read_tool_source(path: Path) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def extract_result_value(result: Any) -> Optional[float]:
    if isinstance(result, dict):
        output = str(result.get("output", ""))
    else:
        output = str(result)

    for token in output.replace(",", " ").split():
        try:
            return float(token)
        except ValueError:
            continue
    return None


def main() -> None:
    timestamp = int(time.time())
    filename = f"random_data_{timestamp}.csv"
    csv_path = ROOT / filename

    create_random_csv(csv_path)

    current_hour = datetime.now().hour
    prompt = (
        f"Calculate the standard deviation of the column 'Value' in `{filename}` "
        f"and multiply it by the current hour."
    )

    before_files = set(p.name for p in TOOLS_DIR.glob("*.py")) if TOOLS_DIR.exists() else set()

    result = evolution_system.process(prompt)

    after_files = set(p.name for p in TOOLS_DIR.glob("*.py")) if TOOLS_DIR.exists() else set()
    new_tool_path = get_new_tool_file(before_files, after_files)
    tool_source = read_tool_source(new_tool_path) if new_tool_path else ""

    uses_csv = "csv" in tool_source
    uses_pandas = "pandas" in tool_source

    truth = compute_truth(csv_path, current_hour)
    predicted = extract_result_value(result)

    verification = {
        "custom_tool_created": new_tool_path is not None,
        "tool_file": str(new_tool_path) if new_tool_path else None,
        "tool_uses_csv_or_pandas": bool(uses_csv or uses_pandas),
        "truth": truth,
        "predicted": predicted,
        "matches_truth": (predicted is not None and abs(predicted - truth) < 1e-6),
    }

    proof = {
        "prompt": prompt,
        "tool_source": tool_source,
        "timestamp": timestamp,
        "final_result": result,
        "verification": verification,
    }

    output_path = ROOT / "evolution_proof.json"
    output_path.write_text(json.dumps(proof, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(proof["verification"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
