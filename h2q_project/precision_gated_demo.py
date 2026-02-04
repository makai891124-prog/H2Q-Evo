"""
DAS Meta-Theory Demonstration
Precision-Gated Executor with Dualistic Verification

This script demonstrates:
1. Metric Decoupling (Axiom III): Separation of discrete logic and continuous manifold
2. Precision-Gated Causality: Entropy-based routing to prevent hallucinations
3. Dualistic Generation (Axiom I): Thesis-Antithesis verification with topological closure
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('precision_gated_demo.log'),
    ]
)

logger = logging.getLogger(__name__)


def demo_precision_gated_executor():
    """Demonstrate PrecisionGatedExecutor with various task types."""
    
    from local_executor import LocalExecutor
    
    logger.info("="*80)
    logger.info("DAS Meta-Theory: Precision-Gated Executor Demo")
    logger.info("="*80)
    
    # Create executor with precision gating enabled
    executor = LocalExecutor(enable_precision_gating=True)
    
    # Test cases covering different entropy profiles
    test_tasks = [
        {
            "name": "High Precision Math Task",
            "task": "Calculate the square root of 144. This is a well-defined mathematical problem with deterministic answer.",
            "expected_state": "particle",
        },
        {
            "name": "High Entropy Reasoning Task",
            "task": "Should we prioritize environmental protection or economic growth? Discuss various perspectives and tradeoffs.",
            "expected_state": "wave",
        },
        {
            "name": "Balanced Logic Task",
            "task": "If all roses are flowers, and some flowers fade, what can we conclude about roses?",
            "expected_state": "coherence",
        },
        {
            "name": "Task with Negation",
            "task": "It is not true that we cannot solve this problem. Explain what this means logically.",
            "expected_state": "wave",
        },
        {
            "name": "Simple Factual Query",
            "task": "What is the capital of France? Paris.",
            "expected_state": "particle",
        },
    ]
    
    results = []
    
    for i, test_case in enumerate(test_tasks, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Test Case {i}: {test_case['name']}")
        logger.info(f"Task: {test_case['task']}")
        logger.info(f"Expected State: {test_case['expected_state']}")
        logger.info(f"{'='*80}")
        
        try:
            # Execute with precision gating
            result = executor.execute(test_case['task'], strategy='auto')
            
            # Extract key information
            state_manifold = result.get('state_manifold', 'unknown')
            entropy = result.get('entropy_metrics', {})
            dualistic = result.get('dualistic_verification', [])
            trace = result.get('execution_trace', [])
            
            logger.info(f"\n[RESULT SUMMARY]")
            logger.info(f"State Manifold: {state_manifold}")
            logger.info(f"Combined Entropy: {entropy.get('combined_entropy', 'N/A'):.4f}")
            logger.info(f"High Precision: {entropy.get('is_high_precision', False)}")
            logger.info(f"Confidence: {result.get('confidence', 0.0):.4f}")
            logger.info(f"Elapsed Time: {result.get('elapsed_time', 0.0):.4f}s")
            
            logger.info(f"\n[ENTROPY BREAKDOWN]")
            logger.info(f"  Logical Entropy:   {entropy.get('logical_entropy', 0.0):.4f}")
            logger.info(f"  Semantic Entropy:  {entropy.get('semantic_entropy', 0.0):.4f}")
            logger.info(f"  Temporal Entropy:  {entropy.get('temporal_entropy', 0.0):.4f}")
            
            logger.info(f"\n[EXECUTION PATH]")
            for step in trace:
                logger.info(f"  -> {step}")
            
            logger.info(f"\n[DUALISTIC VERIFICATION - Axiom I]")
            for j, prop in enumerate(dualistic, 1):
                logger.info(f"\n  Proposition {j}:")
                logger.info(f"    Thesis: {prop['thesis'][:60]}...")
                logger.info(f"    Antithesis: {prop['antithesis'][:60]}...")
                logger.info(f"    P(Thesis): {prop['thesis_confidence']:.4f}")
                logger.info(f"    P(Antithesis): {prop['antithesis_confidence']:.4f}")
                logger.info(f"    Closure Valid: {prop['closure_valid']} (gap: {prop['closure_gap']:.4f})")
            
            logger.info(f"\n[OUTPUT]")
            logger.info(f"Result: {str(result.get('output', 'N/A'))[:100]}...")
            
            results.append({
                'name': test_case['name'],
                'state': state_manifold,
                'entropy': entropy.get('combined_entropy', 0.0),
                'confidence': result.get('confidence', 0.0),
                'closure_valid': all(p.get('closure_valid', False) for p in dualistic),
            })
            
        except Exception as e:
            logger.error(f"Error executing test case: {e}", exc_info=True)
            results.append({
                'name': test_case['name'],
                'error': str(e),
            })
    
    # Summary statistics
    logger.info(f"\n\n{'='*80}")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"{'='*80}")
    
    stats = executor.get_precision_gating_stats()
    logger.info(f"\nPrecision Gating Statistics:")
    logger.info(f"  Total Executions: {stats.get('total_executions', 0)}")
    logger.info(f"  State Distribution: {stats.get('state_distribution', {})}")
    logger.info(f"  Average Entropy: {stats.get('average_entropy', 0.0):.4f}")
    logger.info(f"  Average Confidence: {stats.get('average_confidence', 0.0):.4f}")
    
    logger.info(f"\nTest Results Table:")
    logger.info(f"{'Name':<35} {'State':<12} {'Entropy':<10} {'Confidence':<12} {'Closure':<10}")
    logger.info(f"{'-'*80}")
    
    for result in results:
        if 'error' not in result:
            logger.info(f"{result['name']:<35} {result['state']:<12} "
                       f"{result['entropy']:<10.4f} {result['confidence']:<12.4f} "
                       f"{str(result['closure_valid']):<10}")
        else:
            logger.info(f"{result['name']:<35} ERROR: {result['error']}")
    
    logger.info(f"\n{'='*80}")
    logger.info("DAS Meta-Theory Demo Complete")
    logger.info(f"{'='*80}")


def demo_das_meta_theory_concepts():
    """Explain DAS Meta-Theory concepts with code examples."""
    
    logger.info("\n\n" + "="*80)
    logger.info("DAS Meta-Theory Concepts")
    logger.info("="*80)
    
    logger.info("""
    
1. AXIOM III - Metric Decoupling
   ─────────────────────────────────────────────────────────────────
   Principle: Separate discrete logic (Turing Machine) from continuous 
   manifolds (Quaternion Math).
   
   Implementation:
   - Discrete Layer: Task classification, logical verification, routing decisions
   - Continuous Layer: Quaternion semantic encoding, manifold operations
   
   Code Example:
   ```python
   # Discrete: Classify proposition type
   is_negative = "not" in proposition.lower()  # Turing machine step
   
   # Continuous: Encode to quaternion space
   q_encoded = semantic_anchor * confidence  # Hamilton algebra operation
   q_normalized = q_encoded / ||q_encoded||  # Manifold normalization
   ```


2. PRECISION-GATED CAUSALITY
   ─────────────────────────────────────────────────────────────────
   Principle: Causality only exists when precision is sufficient.
   Entropy-based gating prevents hallucinations.
   
   State Transitions:
   - Wave State (High Entropy):      ↔  Requires orthogonal expansion (CoT)
   - Particle State (Low Entropy):   ↔  Allows direct collapse (Direct Output)
   - Coherence State (Medium):       ↔  Standard verified execution
   
   Entropy Thresholds:
   - Combined Entropy < 0.25   →  PARTICLE (high precision)
   - Combined Entropy > 0.65   →  WAVE (low precision)
   - 0.25 ≤ Entropy ≤ 0.65    →  COHERENCE (balanced)
   
   Entropy Components:
   1. Logical Entropy:   Shannon entropy of discrete propositions
   2. Semantic Entropy:  Uncertainty in quaternion semantic space
   3. Temporal Entropy:  Rate of state transitions over time


3. AXIOM I - DUALISTIC GENERATION
   ─────────────────────────────────────────────────────────────────
   Principle: Verify truth by generating both Thesis (A) and Antithesis (¬A),
   then checking for "Topological Closure" (consistency).
   
   Implementation:
   - Thesis:        Primary interpretation of proposition
   - Antithesis:    Logical negation or contrary interpretation
   - Closure Gap:   |P(A) + P(¬A) - 1.0| (should be ≈ 0)
   - Closure Valid: Topological closure satisfied if gap < tolerance
   
   Verification Logic:
   ```python
   closure_gap = |P(thesis) + P(antithesis) - 1.0|
   is_valid = closure_gap <= 0.05  # Probability distribution must sum to 1
   ```
   
   If closure_valid == False:
   → Indicates hallucination or logical inconsistency
   → System routes to additional verification (CoT, tool use)
   → Prevents direct output with unverified propositions


4. EXECUTION ROUTING (The Middle Layer)
   ─────────────────────────────────────────────────────────────────
   
   Input Task → [Entropy Measurement] → [State Classification]
                                           ↓
                                   ┌───────┴────────┐
                                   │                │
                              WAVE STATE      PARTICLE STATE
                           (High Entropy)   (Low Entropy)
                                   │                │
                          Chain-of-Thought    Direct Output
                          + Tool Use         (Collapsed)
                          + Verification
                                   │                │
                                   └───────┬────────┘
                                           ↓
                                    [Output + Metrics]
    
    
5. HALLUCINATION PREVENTION MECHANISM
   ─────────────────────────────────────────────────────────────────
   
   Traditional LLM:
   Question → LLM → Softmax Collapse → Output
   Problem: No entropy gating, hallucinations possible
   
   DAS Meta-Theory:
   Question → Entropy Measurement
              ↓
           Wave State?  →  Chain-of-Thought (expand reasoning)
              ↓         →  Tool Use (verify facts)
           Particle?    →  Direct Output OK (high confidence)
              ↓
           Dualistic Verification (Thesis-Antithesis)
              ↓
           Topological Closure Check
              ↓
           Output (with confidence and verification metadata)
   
   Key: Don't allow LLM to output if entropy is high!
   
    """)


if __name__ == "__main__":
    try:
        # Show conceptual overview first
        demo_das_meta_theory_concepts()
        
        # Run practical demonstration
        demo_precision_gated_executor()
        
    except Exception as e:
        logger.error(f"Fatal error in demo: {e}", exc_info=True)
        sys.exit(1)
