from __future__ import annotations

import os
import time
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass(frozen=True)
class HardwareProfile:
    torch_version: str
    cpu_count: int
    torch_threads: int
    has_cuda: bool
    has_mps: bool
    cuda_device_name: str | None


@dataclass(frozen=True)
class ComputePlan:
    device: str
    rotor_kernel: str
    compiled: bool
    matmul_precision: str


def default_thread_candidates(cpu_count: int) -> list[int]:
    if cpu_count <= 1:
        return [1]
    base = [1, 2, 4, 6, 8, cpu_count]
    uniq = sorted({max(1, min(cpu_count, int(v))) for v in base})
    return uniq


def parse_thread_candidates(raw: str | None, cpu_count: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return default_thread_candidates(cpu_count)
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        try:
            v = int(tok)
        except ValueError:
            continue
        if v > 0:
            out.append(min(v, cpu_count))
    if not out:
        return default_thread_candidates(cpu_count)
    return sorted(set(out))


def load_compute_plan_cache(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, object]] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, dict):
            out[k] = v
    return out


def save_compute_plan_cache(path: Path, cache: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cache, indent=2, ensure_ascii=True)
    path.write_text(payload, encoding="utf-8")


def make_cache_key(namespace: str, hardware: HardwareProfile, params: dict[str, object]) -> str:
    stable = {
        "ns": namespace,
        "torch": hardware.torch_version,
        "cpu_count": hardware.cpu_count,
        "has_cuda": hardware.has_cuda,
        "has_mps": hardware.has_mps,
        "params": params,
    }
    return json.dumps(stable, sort_keys=True, ensure_ascii=True)


def cache_lookup(cache: dict[str, dict[str, object]], key: str) -> dict[str, object] | None:
    hit = cache.get(key)
    if not isinstance(hit, dict):
        return None
    return hit


def cache_store(cache: dict[str, dict[str, object]], key: str, value: dict[str, object]) -> None:
    cache[key] = value


def detect_hardware_profile() -> HardwareProfile:
    has_cuda = torch.cuda.is_available()
    has_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    cuda_name = torch.cuda.get_device_name(0) if has_cuda else None
    return HardwareProfile(
        torch_version=torch.__version__,
        cpu_count=os.cpu_count() or 1,
        torch_threads=torch.get_num_threads(),
        has_cuda=has_cuda,
        has_mps=has_mps,
        cuda_device_name=cuda_name,
    )


def profile_as_dict(profile: HardwareProfile) -> dict[str, str | int | bool | None]:
    return asdict(profile)


def resolve_device(request: str) -> torch.device:
    key = request.strip().lower()
    if key == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")
    if key == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if key == "mps" and bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return torch.device("mps")
    if key == "cpu":
        return torch.device("cpu")
    return resolve_device("auto")


def apply_runtime_knobs(device: torch.device) -> str:
    # Keep precision hints explicit so experiments are reproducible across hardware.
    if device.type in ("cuda", "mps"):
        torch.set_float32_matmul_precision("high")
        return "high"
    torch.set_float32_matmul_precision("medium")
    return "medium"


def choose_rotor_kernel(
    request: str,
    device: torch.device,
    rank: int,
    rotor_steps: int,
) -> str:
    key = request.strip().lower()
    if key in ("scalar", "staged"):
        return key

    # Auto policy: staged updates are typically better when there are enough rotor steps.
    if rotor_steps >= 8 and rank >= 16 and device.type in ("cpu", "mps", "cuda"):
        return "staged"
    return "scalar"


def try_compile_module(module: torch.nn.Module, enabled: bool) -> tuple[torch.nn.Module, bool]:
    if not enabled:
        return module, False
    if not hasattr(torch, "compile"):
        return module, False
    try:
        return torch.compile(module, mode="reduce-overhead", dynamic=False), True
    except Exception:
        return module, False


def build_nonoverlap_stages(rotor_pairs: torch.Tensor) -> list[list[int]]:
    """
    Group rotor steps into non-overlapping stages, so each stage can update many
    channel pairs in parallel without write conflicts.
    """
    n = int(rotor_pairs.shape[0])
    stages: list[list[int]] = []
    for step in range(n):
        i = int(rotor_pairs[step, 0].item())
        j = int(rotor_pairs[step, 1].item())
        placed = False
        for stage in stages:
            used: set[int] = set()
            for idx in stage:
                used.add(int(rotor_pairs[idx, 0].item()))
                used.add(int(rotor_pairs[idx, 1].item()))
            if i not in used and j not in used:
                stage.append(step)
                placed = True
                break
        if not placed:
            stages.append([step])
    return stages


def pack_stage_steps(stages: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(stages) == 0:
        return (
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )
    max_len = max(len(stage) for stage in stages)
    steps = torch.full((len(stages), max_len), -1, dtype=torch.long, device=device)
    lengths = torch.zeros((len(stages),), dtype=torch.long, device=device)
    for sid, stage in enumerate(stages):
        n = len(stage)
        lengths[sid] = n
        if n > 0:
            steps[sid, :n] = torch.tensor(stage, dtype=torch.long, device=device)
    return steps, lengths


def apply_rotor_scalar(latent: torch.Tensor, rotor_pairs: torch.Tensor, rotor_angles: torch.Tensor) -> torch.Tensor:
    out = latent
    for step in range(rotor_pairs.shape[0]):
        i = int(rotor_pairs[step, 0].item())
        j = int(rotor_pairs[step, 1].item())
        angle = rotor_angles[step]
        c = torch.cos(angle)
        s = torch.sin(angle)
        a = out[:, i].clone()
        b = out[:, j].clone()
        out[:, i] = c * a - s * b
        out[:, j] = s * a + c * b
    return out


def apply_rotor_staged(
    latent: torch.Tensor,
    rotor_pairs: torch.Tensor,
    rotor_angles: torch.Tensor,
    stage_steps: torch.Tensor,
    stage_lengths: torch.Tensor,
) -> torch.Tensor:
    out = latent
    for sid in range(int(stage_lengths.shape[0])):
        n = int(stage_lengths[sid].item())
        if n <= 0:
            continue
        idx = stage_steps[sid, :n]
        pair = rotor_pairs.index_select(0, idx)
        ang = rotor_angles.index_select(0, idx)

        i_idx = pair[:, 0]
        j_idx = pair[:, 1]

        a = out.index_select(1, i_idx)
        b = out.index_select(1, j_idx)

        c = torch.cos(ang).unsqueeze(0)
        s = torch.sin(ang).unsqueeze(0)

        out_i = c * a - s * b
        out_j = s * a + c * b

        out.scatter_(1, i_idx.unsqueeze(0).expand(out_i.shape[0], -1), out_i)
        out.scatter_(1, j_idx.unsqueeze(0).expand(out_j.shape[0], -1), out_j)
    return out


def autotune_callable(
    candidates: dict[str, Callable[[], None]],
    warmup: int = 8,
    repeat: int = 25,
    device: torch.device | None = None,
) -> tuple[str, dict[str, float]]:
    timings: dict[str, float] = {}
    for _, fn in candidates.items():
        for _ in range(warmup):
            fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)

    for name, fn in candidates.items():
        t0 = time.perf_counter()
        for _ in range(repeat):
            fn()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        timings[name] = (t1 - t0) * 1000.0 / repeat

    best = min(timings, key=lambda k: timings[k])
    return best, timings


def autotune_threads_with_matmul(
    thread_candidates: list[int],
    matmul_size: int = 896,
    warmup: int = 2,
    repeat: int = 5,
) -> tuple[int, dict[str, float]]:
    if not thread_candidates:
        return max(1, torch.get_num_threads()), {}

    baseline_threads = torch.get_num_threads()
    timings: dict[str, float] = {}
    best_threads = max(1, baseline_threads)
    best_ms = float("inf")

    try:
        for th in thread_candidates:
            torch.set_num_threads(max(1, int(th)))
            a = torch.randn(matmul_size, matmul_size, dtype=torch.float32)
            b = torch.randn(matmul_size, matmul_size, dtype=torch.float32)
            for _ in range(max(1, warmup)):
                _ = a.matmul(b)
            t0 = time.perf_counter()
            for _ in range(max(1, repeat)):
                _ = a.matmul(b)
            t1 = time.perf_counter()
            ms = ((t1 - t0) * 1000.0) / max(1, repeat)
            timings[f"t{th}"] = float(ms)
            if ms < best_ms:
                best_ms = ms
                best_threads = int(th)
    finally:
        torch.set_num_threads(baseline_threads)

    return max(1, best_threads), timings
