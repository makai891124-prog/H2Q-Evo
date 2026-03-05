"""Tests for the quantum supremacy cross-validation analysis tool.

Covers:
- Amdahl's law fitting
- Time-fold overhead isolation
- Threshold projection logic
- Report generation (structure and completeness)
"""

import math
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAmdahlFit:
    def test_perfect_parallel(self):
        from tools.quantum_supremacy_crossval_analysis import fit_amdahl

        # Perfect parallelism: S(n) = n  → p_serial ≈ 0
        workers = [1, 2, 4]
        speedups = [1.0, 2.0, 4.0]
        result = fit_amdahl(workers, speedups)
        assert result["p_serial"] < 0.05, f"Expected near-zero p_serial, got {result['p_serial']}"
        assert result["max_speedup"] > 10, f"Expected high max speedup, got {result['max_speedup']}"
        assert not result.get("overhead_dominated", False)

    def test_serial_dominated(self):
        from tools.quantum_supremacy_crossval_analysis import fit_amdahl

        # Fully serial: speedup ≈ 1 regardless of workers
        workers = [1, 2, 4]
        speedups = [1.0, 1.0, 1.0]
        result = fit_amdahl(workers, speedups)
        assert result["p_serial"] > 0.9, f"Expected near-1 p_serial, got {result['p_serial']}"
        assert result["max_speedup"] < 1.5
        assert not result.get("overhead_dominated", False)

    def test_overhead_dominated_detected(self):
        from tools.quantum_supremacy_crossval_analysis import fit_amdahl
        import math

        # All speedups < 1: process overhead dominates
        workers = [1, 2, 4]
        speedups = [0.5, 0.6, 0.7]
        result = fit_amdahl(workers, speedups)
        assert result.get("overhead_dominated") is True, "Should detect overhead-dominated regime"
        assert result["p_serial"] == 1.0
        assert result["max_speedup"] == 1.0
        assert math.isnan(result["r2"])

    def test_single_point_returns_safely(self):
        from tools.quantum_supremacy_crossval_analysis import fit_amdahl

        result = fit_amdahl([1], [1.0])
        assert isinstance(result["p_serial"], float)
        assert isinstance(result["max_speedup"], float)


class TestDeterministicModulus:
    def test_bit_width(self):
        from tools.quantum_supremacy_crossval_analysis import _deterministic_modulus

        for bits in [64, 128, 256, 512]:
            n = _deterministic_modulus(bits)
            assert n.bit_length() == bits, (
                f"Expected {bits}-bit number, got {n.bit_length()}"
            )

    def test_odd(self):
        from tools.quantum_supremacy_crossval_analysis import _deterministic_modulus

        for bits in [128, 256]:
            n = _deterministic_modulus(bits)
            assert n % 2 == 1, "Modulus should be odd"

    def test_reproducible(self):
        from tools.quantum_supremacy_crossval_analysis import _deterministic_modulus

        n1 = _deterministic_modulus(256)
        n2 = _deterministic_modulus(256)
        assert n1 == n2, "Should be deterministic across calls"

    def test_different_sizes_differ(self):
        from tools.quantum_supremacy_crossval_analysis import _deterministic_modulus

        n128 = _deterministic_modulus(128)
        n256 = _deterministic_modulus(256)
        assert n128 != n256


class TestTimeFoldOverhead:
    """Verify that snapshot overhead is measurable and small."""

    def test_overhead_fraction_is_small(self):
        from tools.quantum_supremacy_crossval_analysis import (
            CrossValConfig,
            run_scale_crossval,
        )
        from tools.fault_tolerant_rsa_throughput_report import FTParams

        cfg = CrossValConfig(
            worker_counts=[1],
            cv_repeats=1,
            tasks_per_scale={"RSA-100": 50},
            snapshot_overhead_s=0.002,
        )
        ft = FTParams()
        result = run_scale_crossval("RSA-100", digits=100, tasks=50, cfg=cfg, ft=ft)

        # Snapshot overhead should be < 10% of wall time (it's tiny)
        assert result.timefold_overhead_fraction < 0.10, (
            f"Snapshot overhead too large: {result.timefold_overhead_fraction:.4f}"
        )

    def test_pure_le_snap(self):
        """Pure parallel time must be ≤ time with snapshots."""
        from tools.quantum_supremacy_crossval_analysis import (
            CrossValConfig,
            run_scale_crossval,
        )
        from tools.fault_tolerant_rsa_throughput_report import FTParams

        cfg = CrossValConfig(worker_counts=[1], cv_repeats=1,
                              tasks_per_scale={"RSA-100": 50})
        ft = FTParams()
        result = run_scale_crossval("RSA-100", digits=100, tasks=50, cfg=cfg, ft=ft)

        assert result.best_parallel_pure_mean_s <= result.best_parallel_with_snap_mean_s + 1e-6


class TestQuantumComparison:
    """Verify quantum comparison produces sensible values."""

    def test_ft_runtime_positive(self):
        from tools.quantum_supremacy_crossval_analysis import (
            CrossValConfig,
            run_scale_crossval,
        )
        from tools.fault_tolerant_rsa_throughput_report import FTParams

        cfg = CrossValConfig(worker_counts=[1], cv_repeats=1,
                              tasks_per_scale={"RSA-512": 20})
        ft = FTParams()
        result = run_scale_crossval("RSA-512", digits=512, tasks=20, cfg=cfg, ft=ft)

        assert result.ft_quantum_runtime_h > 0
        assert result.classical_pure_h > 0

    def test_classical_faster_than_ft_quantum_small_rsa(self):
        """For small RSA, classical should be much faster than FT quantum."""
        from tools.quantum_supremacy_crossval_analysis import (
            CrossValConfig,
            run_scale_crossval,
        )
        from tools.fault_tolerant_rsa_throughput_report import FTParams

        cfg = CrossValConfig(
            worker_counts=[1], cv_repeats=1,
            tasks_per_scale={"RSA-100": 10},
            p_phys_list=[1e-4],
            factory_counts=[1000],
        )
        ft = FTParams()
        result = run_scale_crossval("RSA-100", digits=100, tasks=10, cfg=cfg, ft=ft)

        # Classical should be faster (ratio < 1.0)
        assert result.quantum_vs_classical_pure < 1.0, (
            "Classical should be faster than FT quantum for RSA-100"
        )


class TestThresholdProjection:
    def test_returns_rows_for_all_projection_digits(self):
        from tools.quantum_supremacy_crossval_analysis import (
            CrossValConfig,
            project_quantum_threshold,
        )
        from tools.fault_tolerant_rsa_throughput_report import FTParams

        cfg = CrossValConfig(
            projection_digits=[100, 512, 2048],
            p_phys_list=[1e-4],
            factory_counts=[1000],
        )
        ft = FTParams()
        result = project_quantum_threshold(cfg, ft)

        assert len(result["rows"]) == 3
        for row in result["rows"]:
            assert row["classical_1000tasks_h"] > 0
            assert row["ft_quantum_best_h"] > 0
            assert isinstance(row["quantum_faster"], bool)

    def test_power_law_exponent_positive(self):
        from tools.quantum_supremacy_crossval_analysis import (
            CrossValConfig,
            project_quantum_threshold,
        )
        from tools.fault_tolerant_rsa_throughput_report import FTParams

        cfg = CrossValConfig(projection_digits=[100, 512], p_phys_list=[1e-4],
                              factory_counts=[1000])
        ft = FTParams()
        result = project_quantum_threshold(cfg, ft)
        # Modexp scales super-linearly with bit size
        assert result["power_law_exponent"] > 1.0


class TestReportGeneration:
    def test_report_contains_key_sections(self, tmp_path):
        from tools.quantum_supremacy_crossval_analysis import (
            CrossValConfig,
            run_analysis,
            build_chinese_report,
            render_plots,
        )

        cfg = CrossValConfig(
            worker_counts=[1, 2],
            cv_repeats=1,
            tasks_per_scale={k: 20 for k in [
                "RSA-100", "RSA-129", "RSA-250",
                "RSA-512", "RSA-768", "RSA-1024", "RSA-2048",
            ]},
            p_phys_list=[1e-4],
            factory_counts=[1000],
            projection_digits=[100, 512, 2048],
        )

        payload = run_analysis(cfg)

        p_s = tmp_path / "s.png"
        p_q = tmp_path / "q.png"
        p_a = tmp_path / "a.png"
        p_t = tmp_path / "t.png"
        render_plots(payload, p_s, p_q, p_a, p_t)

        report = build_chinese_report(
            payload,
            tmp_path / "data.json",
            p_s, p_q, p_a, p_t,
        )

        assert "## 一、实验背景与目标" in report
        assert "## 二、实验方法" in report
        assert "## 三、并行化效率分析结果" in report
        assert "## 四、时间折叠开销分析" in report
        assert "## 五、量子优越性判定" in report
        assert "## 六、量子优越性临界规模投影" in report
        assert "## 七、综合结论" in report
        assert "## 八、附件" in report

    def test_verdict_structure(self):
        from tools.quantum_supremacy_crossval_analysis import CrossValConfig, run_analysis

        cfg = CrossValConfig(
            worker_counts=[1],
            cv_repeats=1,
            tasks_per_scale={k: 10 for k in [
                "RSA-100", "RSA-129", "RSA-250",
                "RSA-512", "RSA-768", "RSA-1024", "RSA-2048",
            ]},
            p_phys_list=[1e-4],
            factory_counts=[1000],
            projection_digits=[100, 512],
        )
        payload = run_analysis(cfg)
        v = payload["verdict"]

        assert "amdahl_p_serial" in v
        assert "amdahl_max_speedup" in v
        assert "avg_timefold_overhead_fraction" in v
        assert "has_real_quantum_advantage_pure" in v
        assert "total_cases" in v
        assert v["total_cases"] == 7
