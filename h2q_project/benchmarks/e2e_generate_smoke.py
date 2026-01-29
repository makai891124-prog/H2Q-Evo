"""Lightweight E2E smoke benchmark for the holomorphic guard + tokenizer/decoder.

Runs a minimal generation loop without external dependencies. The goal is to
exercise the end-to-end contract (prompt -> encode -> holomorphic audit ->
decode) and emit basic timing. Suitable for CI or quick local checks.

Usage:
    python h2q_project/benchmarks/e2e_generate_smoke.py [--prompt "your text"] [--iters N]
"""
import argparse
import time
from typing import Dict, Any, List

import torch

from h2q.core.discrete_decision_engine import get_canonical_dde, LatentConfig
from h2q.core.guards.holomorphic_streaming_middleware import HolomorphicStreamingMiddleware
from h2q.tokenizer_simple import default_tokenizer
from h2q.decoder_simple import default_decoder


def run_smoke(prompt: str = "hello h2q", max_new_tokens: int = 32) -> Dict[str, Any]:
    """Run single generation pass and return metrics."""
    config = LatentConfig(dim=256)
    dde = get_canonical_dde(config=config)
    middleware = HolomorphicStreamingMiddleware(dde=dde, threshold=0.05)

    token_ids = default_tokenizer.encode(prompt, add_specials=True, max_length=256)
    input_tensor = torch.tensor(token_ids, dtype=torch.float32).view(1, -1)

    start = time.perf_counter()
    with torch.no_grad():
        result = middleware.audit_and_execute(
            input_tensor=input_tensor,
            max_steps=max_new_tokens,
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    generated_ids = result.get("generated_token_ids") or token_ids[:max_new_tokens]
    text = default_decoder.decode(default_decoder.trim_at_eos(generated_ids))

    return {
        "prompt": prompt,
        "text": text,
        "fueter_curvature": result.get("fueter_curvature", 0.0),
        "spectral_shift_eta": result.get("spectral_shift", 0.0),
        "latency_ms": elapsed_ms,
        "was_corrected": result.get("was_corrected", False),
    }


def run_benchmark(prompts: List[str], iters: int = 10) -> Dict[str, Any]:
    """Run benchmark over multiple prompts and iterations."""
    latencies: List[float] = []
    curvatures: List[float] = []
    corrections = 0
    total = 0

    for prompt in prompts:
        for _ in range(iters):
            result = run_smoke(prompt)
            latencies.append(result["latency_ms"])
            curvatures.append(result["fueter_curvature"])
            if result["was_corrected"]:
                corrections += 1
            total += 1

    latencies.sort()
    p50_idx = len(latencies) // 2
    p95_idx = int(len(latencies) * 0.95)

    return {
        "total_runs": total,
        "latency_mean_ms": sum(latencies) / len(latencies),
        "latency_p50_ms": latencies[p50_idx],
        "latency_p95_ms": latencies[p95_idx],
        "curvature_mean": sum(curvatures) / len(curvatures),
        "correction_rate": corrections / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="H2Q E2E smoke benchmark")
    parser.add_argument("--prompt", type=str, default="hello h2q", help="Input prompt")
    parser.add_argument("--iters", type=int, default=10, help="Iterations per prompt")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark")
    args = parser.parse_args()

    if args.benchmark:
        prompts = [
            "hello h2q",
            "The quick brown fox",
            "def fibonacci(n):",
            "Explain quaternion mathematics",
        ]
        report = run_benchmark(prompts, iters=args.iters)
        print("[h2q] benchmark report")
        for k, v in report.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        report = run_smoke(args.prompt)
        print("[h2q] e2e smoke report")
        for k, v in report.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
