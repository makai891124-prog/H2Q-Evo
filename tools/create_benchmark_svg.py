#!/usr/bin/env python3
import json
from pathlib import Path

INPUT = Path("benchmark_results/public_benchmark_20260128_211951.json")
OUTPUT = Path("benchmark_results/public_benchmark_20260128_211951.svg")


def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    scores = {
        "mmlu": data["tests"]["mmlu"]["score"] / 100.0,
        "gsm8k": data["tests"]["gsm8k"]["score"] / 100.0,
        "arc": data["tests"]["arc"]["score"] / 100.0,
        "hellaswag": data["tests"]["hellaswag"]["score"] / 100.0,
    }

    W, H = 640, 360
    pad = 40
    bar_w = 100
    gap = 30

    items = list(scores.items())
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
        '<rect width="100%" height="100%" fill="#0b0f14"/>',
        f'<text x="{pad}" y="24" fill="#e6edf3" font-size="16">Public Benchmark Scores</text>',
    ]

    for i, (k, v) in enumerate(items):
        x = pad + i * (bar_w + gap)
        bh = int((H - 2 * pad) * v)
        y = H - pad - bh
        svg.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bh}" fill="#58a6ff"/>')
        svg.append(
            f'<text x="{x + bar_w // 2}" y="{H - pad + 16}" fill="#e6edf3" font-size="12" text-anchor="middle">{k}</text>'
        )
        svg.append(
            f'<text x="{x + bar_w // 2}" y="{y - 6}" fill="#e6edf3" font-size="12" text-anchor="middle">{int(v * 100)}%</text>'
        )

    svg.append("</svg>")
    OUTPUT.write_text("".join(svg), encoding="utf-8")


if __name__ == "__main__":
    main()
