from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class DirectionalAxiomConfig:
    enabled: bool = False
    rank_constraint: int = 8
    horizon_window: int = 16
    stability_threshold: float = 0.80
    projection_error_threshold: float = 0.30
    min_simulation_steps: int = 3
    min_shadow_steps: int = 2
    gate_enforced_min_stability: float = 0.70
    eps: float = 1e-8


class DirectionalAxiomManifoldAdapter:
    """Lightweight rank-constrained directional manifold analyzer."""

    def __init__(self, config: Optional[DirectionalAxiomConfig] = None):
        self.config = config or DirectionalAxiomConfig()
        self._direction_history: Deque[torch.Tensor] = deque(maxlen=max(2, int(self.config.horizon_window)))

    def _normalize_batch(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        return x.float()

    def _project_rank(self, latent_batch: torch.Tensor) -> Dict[str, Any]:
        x = self._normalize_batch(latent_batch)
        if x.numel() == 0:
            return {
                "projected": x,
                "projection_error": 1.0,
                "rank_energy_ratio": 0.0,
                "effective_rank": 0,
                "target_rank": int(self.config.rank_constraint),
            }

        n, d = x.shape
        target_rank = min(max(1, int(self.config.rank_constraint)), n, d)

        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        try:
            _, s, vh = torch.linalg.svd(centered, full_matrices=False)
        except RuntimeError:
            return {
                "projected": x,
                "projection_error": 1.0,
                "rank_energy_ratio": 0.0,
                "effective_rank": 0,
                "target_rank": target_rank,
            }

        basis = vh[:target_rank]
        coeff = centered @ basis.t()
        recon = coeff @ basis + mean

        denom = torch.clamp((x ** 2).mean(), min=self.config.eps)
        proj_err = torch.clamp(((x - recon) ** 2).mean() / denom, min=0.0, max=1.0).item()

        energy = s ** 2
        energy_total = torch.clamp(energy.sum(), min=self.config.eps)
        rank_energy_ratio = torch.clamp(energy[:target_rank].sum() / energy_total, min=0.0, max=1.0).item()

        return {
            "projected": coeff,
            "projection_error": float(proj_err),
            "rank_energy_ratio": float(rank_energy_ratio),
            "effective_rank": int(target_rank),
            "target_rank": int(target_rank),
        }

    def _direction_stability(self) -> float:
        if len(self._direction_history) < 2:
            return 1.0

        angles = []
        hist = list(self._direction_history)
        for i in range(1, len(hist)):
            cos = F.cosine_similarity(hist[i - 1], hist[i], dim=0, eps=self.config.eps)
            cos_v = torch.clamp(cos, min=-1.0, max=1.0)
            angle_norm = float(torch.arccos(cos_v).item() / math.pi)
            angles.append(angle_norm)

        mean_angle = sum(angles) / max(1, len(angles))
        return float(max(0.0, min(1.0, 1.0 - mean_angle)))

    def analyze(self, latent_batch: torch.Tensor, generation: int) -> Dict[str, Any]:
        projection = self._project_rank(latent_batch)
        projected = projection["projected"]

        if projected.numel() > 0:
            directions = F.normalize(projected, p=2, dim=-1, eps=self.config.eps)
            direction_mean = F.normalize(directions.mean(dim=0), p=2, dim=0, eps=self.config.eps)
            self._direction_history.append(direction_mean.detach().cpu())

        stability = self._direction_stability()
        rolling_pass = (
            stability >= float(self.config.stability_threshold)
            and float(projection["projection_error"]) <= float(self.config.projection_error_threshold)
        )

        return {
            "generation": int(generation),
            "rank_constraint": int(self.config.rank_constraint),
            "effective_rank": int(projection["effective_rank"]),
            "rank_energy_ratio": float(projection["rank_energy_ratio"]),
            "projection_error": float(projection["projection_error"]),
            "direction_stability": float(stability),
            "rolling_horizon_window": int(self.config.horizon_window),
            "rolling_horizon_pass": bool(rolling_pass),
            "thresholds": {
                "stability": float(self.config.stability_threshold),
                "projection_error": float(self.config.projection_error_threshold),
            },
        }


class DirectionalColdStartController:
    """Three-phase controller: simulation -> shadow -> gate_enforced."""

    PHASE_SIMULATION = "simulation"
    PHASE_SHADOW = "shadow"
    PHASE_GATE_ENFORCED = "gate_enforced"

    def __init__(self, config: DirectionalAxiomConfig):
        self.config = config
        self.phase = self.PHASE_SIMULATION
        self.total_steps = 0
        self.shadow_steps = 0

    def update(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        self.total_steps += 1
        rolling_ok = bool(metrics.get("rolling_horizon_pass", False))
        stability = float(metrics.get("direction_stability", 0.0))
        transition = "hold"

        if self.phase == self.PHASE_SIMULATION:
            if self.total_steps >= int(self.config.min_simulation_steps) and rolling_ok:
                self.phase = self.PHASE_SHADOW
                self.shadow_steps = 0
                transition = "simulation_to_shadow"
        elif self.phase == self.PHASE_SHADOW:
            self.shadow_steps += 1
            if not rolling_ok:
                self.phase = self.PHASE_SIMULATION
                self.shadow_steps = 0
                transition = "shadow_to_simulation"
            elif self.shadow_steps >= int(self.config.min_shadow_steps):
                self.phase = self.PHASE_GATE_ENFORCED
                transition = "shadow_to_gate_enforced"
        elif self.phase == self.PHASE_GATE_ENFORCED:
            if stability < float(self.config.gate_enforced_min_stability):
                self.phase = self.PHASE_SHADOW
                self.shadow_steps = 0
                transition = "gate_enforced_to_shadow"

        return {
            "phase": self.phase,
            "transition": transition,
            "total_steps": int(self.total_steps),
            "shadow_steps": int(self.shadow_steps),
        }

    def snapshot(self) -> Dict[str, Any]:
        out = asdict(self.config)
        out.update(
            {
                "phase": self.phase,
                "total_steps": int(self.total_steps),
                "shadow_steps": int(self.shadow_steps),
            }
        )
        return out
