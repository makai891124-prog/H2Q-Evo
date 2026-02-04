"""
Unit tests for PrecisionGatedExecutor and DAS Meta-Theory components.
"""

import unittest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Assuming imports work from h2q_project
from precision_gated_executor import (
    EntropyMetrics,
    StateManifold,
    DualProposition,
    ContinuousManifoldEncoder,
    DiscreteLogicVerifier,
    PrecisionGatedExecutor,
)


class TestEntropyMetrics(unittest.TestCase):
    """Test entropy measurement and state manifold classification."""
    
    def test_high_precision_classification(self):
        """Test that low entropy is classified as particle state."""
        metrics = EntropyMetrics(
            logical_entropy=0.1,
            semantic_entropy=0.05,
            temporal_entropy=0.02,
            combined_entropy=0.05,
        )
        
        self.assertTrue(metrics.is_high_precision(threshold=0.3))
        self.assertEqual(metrics.get_manifold_state(), StateManifold.PARTICLE)
    
    def test_low_precision_classification(self):
        """Test that high entropy is classified as wave state."""
        metrics = EntropyMetrics(
            logical_entropy=0.8,
            semantic_entropy=0.7,
            temporal_entropy=0.6,
            combined_entropy=0.7,
        )
        
        self.assertFalse(metrics.is_high_precision(threshold=0.3))
        self.assertEqual(metrics.get_manifold_state(), StateManifold.WAVE)
    
    def test_coherence_state(self):
        """Test balanced entropy classified as coherence."""
        metrics = EntropyMetrics(
            logical_entropy=0.4,
            semantic_entropy=0.45,
            temporal_entropy=0.42,
            combined_entropy=0.42,
        )
        
        self.assertEqual(metrics.get_manifold_state(), StateManifold.COHERENCE)


class TestDualProposition(unittest.TestCase):
    """Test dualistic verification and topological closure."""
    
    def test_valid_topological_closure(self):
        """Test that valid probability distributions are recognized."""
        prop = DualProposition(
            thesis="A is true",
            antithesis="A is false",
            thesis_confidence=0.6,
            antithesis_confidence=0.4,
        )
        
        is_valid = prop.verify_closure(tolerance=0.05)
        
        self.assertTrue(is_valid)
        self.assertLess(prop.closure_gap, 0.05)
    
    def test_invalid_topological_closure(self):
        """Test that invalid probability distributions are caught."""
        prop = DualProposition(
            thesis="A is true",
            antithesis="A is false",
            thesis_confidence=0.7,
            antithesis_confidence=0.7,  # sum > 1.0
        )
        
        is_valid = prop.verify_closure(tolerance=0.05)
        
        self.assertFalse(is_valid)
        self.assertGreater(prop.closure_gap, 0.05)
    
    def test_closure_gap_calculation(self):
        """Test closure gap calculation."""
        prop = DualProposition(
            thesis="A",
            antithesis="not A",
            thesis_confidence=0.55,
            antithesis_confidence=0.45,
        )
        
        prop.verify_closure()
        
        expected_gap = abs(0.55 + 0.45 - 1.0)
        self.assertAlmostEqual(prop.closure_gap, expected_gap, places=10)


class TestContinuousManifoldEncoder(unittest.TestCase):
    """Test quaternion semantic encoding."""
    
    def setUp(self):
        self.encoder = ContinuousManifoldEncoder()
    
    def test_proposition_encoding_returns_unit_quaternion(self):
        """Test that encoded propositions are normalized."""
        q = self.encoder.encode_proposition("This is a test proposition")
        
        norm = np.linalg.norm(q)
        self.assertAlmostEqual(norm, 1.0, places=6)
    
    def test_affirmative_proposition(self):
        """Test encoding of affirmative propositions."""
        q_affirmative = self.encoder.encode_proposition("The answer is yes")
        q_negative = self.encoder.encode_proposition("The answer is not yes")
        
        # Should be different in quaternion space
        distance = self.encoder.quaternion_distance(q_affirmative, q_negative)
        self.assertGreater(distance, 0.1)
    
    def test_uncertain_proposition(self):
        """Test encoding of uncertain propositions."""
        q_certain = self.encoder.encode_proposition("The answer is definitely yes")
        q_uncertain = self.encoder.encode_proposition("The answer might be yes")
        
        distance = self.encoder.quaternion_distance(q_certain, q_uncertain)
        self.assertGreater(distance, 0.0)
    
    def test_confidence_extraction(self):
        """Test confidence extraction from text."""
        conf_certain = self.encoder._extract_confidence("definitely true")
        conf_uncertain = self.encoder._extract_confidence("possibly true")
        conf_unlikely = self.encoder._extract_confidence("unlikely to happen")
        
        self.assertGreater(conf_certain, conf_uncertain)
        self.assertGreater(conf_uncertain, conf_unlikely)
    
    def test_quaternion_distance_symmetry(self):
        """Test that quaternion distance is symmetric."""
        q1 = self.encoder.encode_proposition("Proposition A")
        q2 = self.encoder.encode_proposition("Proposition B")
        
        dist_1_2 = self.encoder.quaternion_distance(q1, q2)
        dist_2_1 = self.encoder.quaternion_distance(q2, q1)
        
        self.assertAlmostEqual(dist_1_2, dist_2_1, places=6)
    
    def test_proposition_cache(self):
        """Test that propositions are cached."""
        prop = "Test proposition for caching"
        
        q1 = self.encoder.encode_proposition(prop)
        q2 = self.encoder.encode_proposition(prop)
        
        # Should be identical (same object from cache)
        np.testing.assert_array_equal(q1, q2)
        self.assertIn(prop, self.encoder._proposition_cache)


class TestDiscreteLogicVerifier(unittest.TestCase):
    """Test discrete logic verification."""
    
    def setUp(self):
        self.verifier = DiscreteLogicVerifier()
    
    def test_explicit_contradiction_detection(self):
        """Test detection of explicit logical contradictions."""
        thesis = "The statement is true"
        antithesis = "The statement is not true"
        
        is_contradiction = self.verifier.verify_contradiction(thesis, antithesis)
        
        self.assertTrue(is_contradiction)
    
    def test_non_contradiction_detection(self):
        """Test that non-contradictory statements are recognized."""
        statement1 = "The sun is bright"
        statement2 = "The sky is blue"
        
        is_contradiction = self.verifier.verify_contradiction(statement1, statement2)
        
        self.assertFalse(is_contradiction)
    
    def test_negation_pattern_matching(self):
        """Test detection of various negation patterns."""
        affirmative = "Roses are red"
        negative_not = "Not roses are red"
        negative_no = "No roses are red"
        
        self.assertTrue(self.verifier.verify_contradiction(affirmative, negative_not))
        self.assertTrue(self.verifier.verify_contradiction(affirmative, negative_no))
    
    def test_logical_consistency_check(self):
        """Test consistency verification across multiple propositions."""
        propositions = [
            "If A then B",
            "A is true",
            "Therefore B is true",
        ]
        
        is_consistent, conflicts = self.verifier.verify_logical_consistency(propositions)
        
        # These statements should be logically consistent
        self.assertTrue(is_consistent)
        self.assertEqual(len(conflicts), 0)
    
    def test_logical_inconsistency_detection(self):
        """Test detection of logical inconsistencies."""
        propositions = [
            "The cat is on the table",
            "The cat is not on the table",
        ]
        
        is_consistent, conflicts = self.verifier.verify_logical_consistency(propositions)
        
        self.assertFalse(is_consistent)
        self.assertGreater(len(conflicts), 0)


class TestPrecisionGatedExecutor(unittest.TestCase):
    """Test the main PrecisionGatedExecutor."""
    
    def setUp(self):
        self.mock_executor = Mock()
        self.mock_executor.execute = Mock(return_value={
            "output": "Test output",
            "confidence": 0.8,
        })
        
        self.executor = PrecisionGatedExecutor(
            base_executor=self.mock_executor,
            enable_cot=True,
        )
    
    def test_entropy_measurement(self):
        """Test entropy measurement functionality."""
        task = "Calculate the sum of 2 and 3"
        
        metrics = self.executor._measure_entropy(task)
        
        self.assertIsInstance(metrics, EntropyMetrics)
        self.assertGreaterEqual(metrics.logical_entropy, 0.0)
        self.assertGreaterEqual(metrics.semantic_entropy, 0.0)
        self.assertGreaterEqual(metrics.temporal_entropy, 0.0)
        self.assertLessEqual(metrics.combined_entropy, 1.0)
    
    def test_high_precision_routing(self):
        """Test that high precision tasks route to direct execution."""
        task = "2 + 2 = 4"  # Very clear, low entropy task
        
        result = self.executor.execute_with_precision_gating(task)
        
        # Check that result contains expected fields
        self.assertIn("output", result)
        self.assertIn("state_manifold", result)
        self.assertIn("probe", result)
    
    def test_dualistic_proposition_generation(self):
        """Test generation of thesis-antithesis pairs."""
        task = "Is the earth round?"
        
        propositions = self.executor._generate_dualistic_propositions(task)
        
        self.assertGreater(len(propositions), 0)
        for prop in propositions:
            self.assertIsInstance(prop, DualProposition)
            self.assertIsNotNone(prop.thesis)
            self.assertIsNotNone(prop.antithesis)
    
    def test_task_decomposition(self):
        """Test task decomposition for chain-of-thought."""
        task = "First solve A, then solve B, and finally solve C"
        
        subtasks = self.executor._decompose_task(task)
        
        self.assertGreater(len(subtasks), 1)
        self.assertLessEqual(len(subtasks), 3)  # max_subtasks default
    
    def test_execution_statistics(self):
        """Test collection of execution statistics."""
        # Run several executions
        for i in range(3):
            self.executor.execute_with_precision_gating(f"Task {i}")
        
        stats = self.executor.get_execution_statistics()
        
        self.assertEqual(stats["total_executions"], 3)
        self.assertIn("state_distribution", stats)
        self.assertIn("average_entropy", stats)
        self.assertIn("average_confidence", stats)
    
    def test_execution_trace_recording(self):
        """Test that execution traces are properly recorded."""
        task = "Test task for tracing"
        
        result = self.executor.execute_with_precision_gating(task)
        
        self.assertIn("execution_trace", result)
        self.assertGreater(len(result["execution_trace"]), 0)
        
        # Check for expected trace steps
        trace = result["execution_trace"]
        self.assertIn("STEP_0_PROBE_GENERATION", trace)
        self.assertIn("STEP_1_PRECISION_GATE", trace)


class TestIntegrationWithLocalExecutor(unittest.TestCase):
    """Test integration with LocalExecutor."""
    
    @patch('local_executor.LearningLoop')
    @patch('local_executor.StrategyManager')
    @patch('local_executor.FeedbackHandler')
    def test_local_executor_with_precision_gating(self, mock_feedback, mock_strategy, mock_learning):
        """Test LocalExecutor initialization with precision gating."""
        from local_executor import LocalExecutor
        
        # This test verifies that LocalExecutor properly integrates PrecisionGatedExecutor
        executor = LocalExecutor(enable_precision_gating=True)
        
        self.assertIsNotNone(executor.precision_gated_executor)
        self.assertTrue(executor.enable_precision_gating)
    
    @patch('local_executor.LearningLoop')
    @patch('local_executor.StrategyManager')
    @patch('local_executor.FeedbackHandler')
    def test_local_executor_precision_gating_stats(self, mock_feedback, mock_strategy, mock_learning):
        """Test stats retrieval from LocalExecutor."""
        from local_executor import LocalExecutor
        
        executor = LocalExecutor(enable_precision_gating=True)
        stats = executor.get_precision_gating_stats()
        
        self.assertIn("enabled", stats)
        self.assertTrue(stats["enabled"])


class TestDASMetaTheoryAxioms(unittest.TestCase):
    """Test DAS Meta-Theory axioms and principles."""
    
    def test_axiom_iii_metric_decoupling(self):
        """Test Axiom III: Metric Decoupling (Discrete vs Continuous)."""
        encoder = ContinuousManifoldEncoder()
        verifier = DiscreteLogicVerifier()
        
        # Discrete layer: Logic verification
        contradicts = verifier.verify_contradiction("A is true", "A is not true")
        self.assertTrue(contradicts)
        
        # Continuous layer: Semantic encoding
        q1 = encoder.encode_proposition("A is true")
        q2 = encoder.encode_proposition("A is not true")
        
        # Different in quaternion space
        distance = encoder.quaternion_distance(q1, q2)
        self.assertGreater(distance, 0.0)
    
    def test_axiom_i_dualistic_generation(self):
        """Test Axiom I: Dualistic Generation with Topological Closure."""
        executor = PrecisionGatedExecutor()
        
        task = "Is AI safe?"
        props = executor._generate_dualistic_propositions(task)
        
        for prop in props:
            # Each must have both thesis and antithesis
            self.assertIsNotNone(prop.thesis)
            self.assertIsNotNone(prop.antithesis)
            
            # Topological closure should be verified
            prop.verify_closure()
            self.assertLess(prop.closure_gap, 0.5)  # Reasonable gap
    
    def test_precision_gated_causality(self):
        """Test Precision-Gated Causality principle."""
        executor = PrecisionGatedExecutor()
        
        # Low entropy task (math)
        low_entropy_task = "What is 2+2?"
        low_entropy_metrics = executor._measure_entropy(low_entropy_task)
        
        # High entropy task (opinion)
        high_entropy_task = "What is the meaning of life, considering philosophy, science, and culture?"
        high_entropy_metrics = executor._measure_entropy(high_entropy_task)
        
        # High entropy should be greater than low
        self.assertGreater(
            high_entropy_metrics.combined_entropy,
            low_entropy_metrics.combined_entropy
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
