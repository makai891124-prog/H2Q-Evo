import numpy as np
import pytest

from h2q_project.das_gqs.backends import available_backends
from h2q_project.das_gqs.core import Bivector, Vector, generate_rotor, sandwich_rotate


def test_numpy_backend_unit_norm_preserved():
    rotor = generate_rotor(Bivector(np.array([0.3, -0.7, 0.2])), angle=1.234)
    v = Vector(np.array([1.1, -0.2, 0.9]))
    out = sandwich_rotate(v, rotor, backend="numpy").value
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-9)


@pytest.mark.skipif("clifford" not in available_backends(), reason="clifford package unavailable")
def test_numpy_and_clifford_consistent():
    rng = np.random.default_rng(2026)
    max_l2 = 0.0
    for _ in range(16):
        axis = rng.normal(size=3)
        angle = float(rng.uniform(-2.0 * np.pi, 2.0 * np.pi))
        vec = rng.normal(size=3)

        rotor = generate_rotor(Bivector(axis), angle=angle)
        v = Vector(vec)
        out_np = sandwich_rotate(v, rotor, backend="numpy").value
        out_cf = sandwich_rotate(v, rotor, backend="clifford").value
        max_l2 = max(max_l2, float(np.linalg.norm(out_np - out_cf)))

    assert max_l2 < 1e-7
