from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


EPS = 1e-9


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / n


def _as_np3(value: np.ndarray | Tuple[float, float, float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


@dataclass(frozen=True)
class BackendRotorSpec:
    scalar: float
    bivector_e23_e31_e12: np.ndarray

    @property
    def axis(self) -> np.ndarray:
        return _normalize(_as_np3(self.bivector_e23_e31_e12))

    @property
    def angle(self) -> float:
        biv_norm = np.linalg.norm(self.bivector_e23_e31_e12)
        return 2.0 * float(np.arctan2(biv_norm, self.scalar))


class NumpyBackend:
    name = "numpy"

    def rotate(self, vector: np.ndarray, rotor: BackendRotorSpec) -> np.ndarray:
        axis = rotor.axis
        theta = rotor.angle
        x = _normalize(_as_np3(vector))
        c = np.cos(theta)
        s = np.sin(theta)
        rotated = c * x + s * np.cross(axis, x) + (1.0 - c) * np.dot(axis, x) * axis
        return _normalize(rotated)


class CliffordBackend:
    name = "clifford"

    def __init__(self) -> None:
        try:
            from clifford.g3 import blades, layout  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "clifford backend requested but package 'clifford' is not installed"
            ) from exc

        self._layout = layout
        self._blades = blades
        self._e1 = self._blades["e1"]
        self._e2 = self._blades["e2"]
        self._e3 = self._blades["e3"]
        self._e12 = self._blades["e12"]
        self._e13 = self._blades["e13"]
        self._e23 = self._blades["e23"]

    def _vec(self, v: np.ndarray):
        x = _as_np3(v)
        return x[0] * self._e1 + x[1] * self._e2 + x[2] * self._e3

    def _bivector(self, b: np.ndarray):
        # Input basis order is (e23, e31, e12). In clifford basis e31 = -e13.
        bb = _as_np3(b)
        return bb[0] * self._e23 + bb[1] * (-self._e13) + bb[2] * self._e12

    def rotate(self, vector: np.ndarray, rotor: BackendRotorSpec) -> np.ndarray:
        v = self._vec(_normalize(_as_np3(vector)))
        b_unit = _normalize(_as_np3(rotor.bivector_e23_e31_e12))
        B = self._bivector(b_unit)
        half = 0.5 * rotor.angle
        R = np.cos(half) - B * np.sin(half)
        rotated = R * v * ~R
        out = np.array(
            [
                float((rotated | self._e1)[()]),
                float((rotated | self._e2)[()]),
                float((rotated | self._e3)[()]),
            ],
            dtype=float,
        )
        return _normalize(out)


def get_backend(name: str):
    key = name.strip().lower()
    if key == "numpy":
        return NumpyBackend()
    if key == "clifford":
        return CliffordBackend()
    raise ValueError(f"Unsupported backend: {name}")


def available_backends() -> list[str]:
    out = ["numpy"]
    try:
        CliffordBackend()
        out.append("clifford")
    except ModuleNotFoundError:
        pass
    return out
