from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .backends import BackendRotorSpec, get_backend


EPS = 1e-9


def _as_np3(value: np.ndarray | Tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(3)
    return arr


def _normalize(value: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(value)
    if n < EPS:
        raise ValueError("Zero-length vector cannot be normalized")
    return value / n


@dataclass(frozen=True)
class Scalar:
    value: float


@dataclass(frozen=True)
class Vector:
    value: np.ndarray

    def normalized(self) -> "Vector":
        return Vector(_normalize(_as_np3(self.value)))


@dataclass(frozen=True)
class Bivector:
    # Components in basis (e23, e31, e12). This is dual to a 3D axis vector.
    value: np.ndarray

    def normalized(self) -> "Bivector":
        return Bivector(_normalize(_as_np3(self.value)))


@dataclass(frozen=True)
class Pseudoscalar:
    value: float = 1.0


@dataclass(frozen=True)
class Rotor:
    # R = s + B, where B is bivector part.
    scalar: float
    bivector: Bivector

    def reverse(self) -> "Rotor":
        return Rotor(self.scalar, Bivector(-self.bivector.value))

    def axis(self) -> np.ndarray:
        # In G3, unit bivector is dual to unit axis.
        return _normalize(self.bivector.value)

    def angle(self) -> float:
        s_norm = np.linalg.norm(self.bivector.value)
        return 2.0 * np.arctan2(s_norm, self.scalar)


class G3:
    e1 = Vector(np.array([1.0, 0.0, 0.0]))
    e2 = Vector(np.array([0.0, 1.0, 0.0]))
    e3 = Vector(np.array([0.0, 0.0, 1.0]))

    # Bivector basis in (e23, e31, e12)
    e23 = Bivector(np.array([1.0, 0.0, 0.0]))
    e31 = Bivector(np.array([0.0, 1.0, 0.0]))
    e12 = Bivector(np.array([0.0, 0.0, 1.0]))

    I = Pseudoscalar(1.0)


def generate_rotor(bivector_plane: Bivector, angle: float) -> Rotor:
    b = bivector_plane.normalized().value
    half = 0.5 * angle
    scalar = float(np.cos(half))
    # R = cos(theta/2) - B sin(theta/2)
    bivector = Bivector(-b * np.sin(half))
    return Rotor(scalar=scalar, bivector=bivector)


def sandwich_rotate(v: Vector, rotor: Rotor, backend: str = "numpy") -> Vector:
    impl = get_backend(backend)
    rotated = impl.rotate(
        vector=v.normalized().value,
        rotor=BackendRotorSpec(
            scalar=rotor.scalar,
            bivector_e23_e31_e12=rotor.bivector.value,
        ),
    )
    return Vector(_normalize(rotated))


def assert_reversible(
    v_old: Vector,
    rotor: Rotor,
    atol: float = 1e-8,
    backend: str = "numpy",
) -> Dict[str, float]:
    v_new = sandwich_rotate(v_old, rotor, backend=backend)
    recovered = sandwich_rotate(v_new, rotor.reverse(), backend=backend)
    err = float(np.linalg.norm(recovered.value - v_old.normalized().value))
    if not np.allclose(recovered.value, v_old.normalized().value, atol=atol):
        raise AssertionError(f"Reversibility failed, L2 error={err:.3e}")
    return {"reversibility_l2_error": err}


class EntangledPair:
    """
    DAS geometric entanglement surrogate:
    two separated Bloch/Riemann-sphere poles are phase-locked by inversion symmetry.
    """

    def __init__(self, a_state: Vector | None = None):
        self.a_state = (a_state or G3.e3).normalized()
        self.b_state = Vector(-self.a_state.value)

    def apply_global_correlated_rotor(self, rotor: Rotor, backend: str = "numpy") -> None:
        # A evolves by R v ~R, B stays strictly inversion-locked: v_B = -v_A.
        self.a_state = sandwich_rotate(self.a_state, rotor, backend=backend)
        self.b_state = Vector(-self.a_state.value)

    def poles(self) -> Tuple[Vector, Vector]:
        return self.a_state, self.b_state


def measure_projection(state: Vector, axis: Vector) -> Tuple[float, int, Vector]:
    v = state.normalized().value
    m = axis.normalized().value
    p = float(np.dot(v, m))
    outcome = 1 if p >= 0.0 else -1
    collapsed = Vector(m if outcome > 0 else -m)
    return p, outcome, collapsed


def geometric_correlation(axis_a: Vector, axis_b: Vector) -> float:
    # Singlet-equivalent geometric correlation in G3: E(a,b) = -a·b.
    a = axis_a.normalized().value
    b = axis_b.normalized().value
    return float(-np.dot(a, b))
