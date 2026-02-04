"""
DAS Meta-Theory Integration Example with LocalExecutor

This example demonstrates how to use the PrecisionGatedExecutor
integrated with LocalExecutor for real-world use cases.
"""

def example_1_basic_usage():
    """Example 1: Basic usage with automatic precision gating."""
    print("\n" + "="*70)
    print("Example 1: Basic Usage with Precision Gating")
    print("="*70)
    
    # Import (would be: from h2q_project.local_executor import LocalExecutor)
    from h2q_project.local_executor import LocalExecutor
    
    # Create executor with precision gating enabled
    executor = LocalExecutor(enable_precision_gating=True)
    
    # Test cases with different entropy profiles
    test_cases = [
        {
            "task": "What is 2 + 2?",
            "description": "Low entropy (deterministic)"
        },
        {
            "task": "Should we prioritize AI safety or capabilities development?",
            "description": "High entropy (opinion-based)"
        },
        {
            "task": "Paris is the capital of France. Where is Paris?",
            "description": "Factual query"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Task: {test['task'][:50]}...")
        
        # Execute with precision gating
        result = executor.execute(test['task'])
        
        # Display key results
        print(f"  State: {result.get('state_manifold', 'unknown')}")
        print(f"  Entropy: {result.get('entropy_metrics', {}).get('combined_entropy', 0):.4f}")
        print(f"  Confidence: {result.get('confidence', 0):.2f}")
        print(f"  Output: {str(result.get('output', 'N/A'))[:60]}...")


def example_2_entropy_analysis():
    """Example 2: Analyzing entropy in different task types."""
    print("\n" + "="*70)
    print("Example 2: Entropy Analysis Across Task Types")
    print("="*70)
    
    from h2q_project.local_executor import LocalExecutor
    
    executor = LocalExecutor(enable_precision_gating=True)
    
    tasks = {
        "Math": "Calculate the derivative of x^2",
        "Logic": "If A implies B and B is true, what about A?",
        "Opinion": "What is the best programming language?",
        "Factual": "What year was Python created?",
        "Complex": "Explain quantum computing, considering wave functions, entanglement, and practical applications",
    }
    
    results = []
    
    for task_type, task in tasks.items():
        result = executor.execute(task)
        entropy_val = result.get('entropy_metrics', {}).get('combined_entropy', 0)
        state = result.get('state_manifold', 'unknown')
        results.append((task_type, entropy_val, state))
        
        print(f"\n{task_type:12} | Entropy: {entropy_val:.4f} | State: {state:10}")
    
    # Analyze pattern
    print("\n" + "-"*70)
    print("Entropy Pattern Analysis:")
    
    particle_tasks = [t[0] for t in results if t[2] == 'particle']
    wave_tasks = [t[0] for t in results if t[2] == 'wave']
    
    if particle_tasks:
        print(f"  High Precision (Particle): {', '.join(particle_tasks)}")
    if wave_tasks:
        print(f"  Low Precision (Wave):      {', '.join(wave_tasks)}")


def example_3_dualistic_verification():
    """Example 3: Understanding dualistic verification."""
    print("\n" + "="*70)
    print("Example 3: Dualistic Verification Details")
    print("="*70)
    
    from h2q_project.local_executor import LocalExecutor
    
    executor = LocalExecutor(enable_precision_gating=True)
    
    # Complex task requiring dual verification
    task = "Is artificial general intelligence achievable?"
    
    print(f"\nTask: {task}")
    result = executor.execute(task)
    
    # Show dualistic verification details
    print("\nDualistic Propositions (Axiom I):")
    print("-" * 70)
    
    for i, prop in enumerate(result.get('dualistic_verification', []), 1):
        print(f"\nProposition {i}:")
        print(f"  Thesis:            {prop['thesis']}")
        print(f"  Antithesis:        {prop['antithesis']}")
        print(f"  P(Thesis):         {prop['thesis_confidence']:.4f}")
        print(f"  P(Antithesis):     {prop['antithesis_confidence']:.4f}")
        print(f"  Sum of Probs:      {prop['thesis_confidence'] + prop['antithesis_confidence']:.4f}")
        print(f"  Closure Valid:     {prop['closure_valid']}")
        print(f"  Closure Gap:       {prop['closure_gap']:.6f}")
        
        if prop['closure_valid']:
            print(f"  Status:            ✓ Logically consistent")
        else:
            print(f"  Status:            ✗ Potential hallucination detected")


def example_4_execution_tracing():
    """Example 4: Understanding execution flow with tracing."""
    print("\n" + "="*70)
    print("Example 4: Execution Tracing")
    print("="*70)
    
    from h2q_project.local_executor import LocalExecutor
    
    executor = LocalExecutor(enable_precision_gating=True)
    
    # Simple task
    task = "Calculate factorial of 5"
    result = executor.execute(task)
    
    print(f"\nTask: {task}")
    print("\nExecution Path:")
    
    for i, step in enumerate(result.get('execution_trace', []), 1):
        indent = "  " * (step.count("_") // 5)
        print(f"{indent}({i}) {step}")
    
    # Interpret the path
    trace = result.get('execution_trace', [])
    if 'ROUTE_DIRECT_OUTPUT' in trace:
        print("\n→ Conclusion: High precision allowed direct output (Particle state)")
    elif 'ROUTE_CHAIN_OF_THOUGHT' in trace:
        print("\n→ Conclusion: Low precision required chain-of-thought reasoning (Wave state)")
    elif 'ROUTE_STANDARD_VERIFIED' in trace:
        print("\n→ Conclusion: Balanced precision with standard verification (Coherence state)")


def example_5_statistics_and_monitoring():
    """Example 5: Statistics and monitoring."""
    print("\n" + "="*70)
    print("Example 5: Statistics and Monitoring")
    print("="*70)
    
    from h2q_project.local_executor import LocalExecutor
    
    # Create executor
    executor = LocalExecutor(enable_precision_gating=True)
    
    # Run multiple tasks
    tasks = [
        "2+2=?",
        "What is consciousness?",
        "Solve x^2 + 5x + 6 = 0",
        "How should society balance innovation and safety?",
        "What is the boiling point of water?",
    ]
    
    print(f"\nExecuting {len(tasks)} tasks for statistical analysis...")
    
    for task in tasks:
        executor.execute(task)
    
    # Get statistics
    stats = executor.get_precision_gating_stats()
    
    print("\nExecution Statistics:")
    print(f"  Total Executions:     {stats.get('total_executions', 0)}")
    print(f"  Average Entropy:      {stats.get('average_entropy', 0):.4f}")
    print(f"  Average Confidence:   {stats.get('average_confidence', 0):.4f}")
    
    print("\nState Distribution:")
    for state, count in stats.get('state_distribution', {}).items():
        percentage = (count / stats.get('total_executions', 1)) * 100
        bar = "█" * int(percentage / 5)
        print(f"  {state:10}: {count:2} tasks ({percentage:5.1f}%) {bar}")


def example_6_comparison_with_without_gating():
    """Example 6: Comparing results with and without precision gating."""
    print("\n" + "="*70)
    print("Example 6: Impact of Precision Gating")
    print("="*70)
    
    from h2q_project.local_executor import LocalExecutor
    
    task = "Is this statement true: This statement is false."
    
    print(f"\nTask: {task}")
    print("(Classic paradox to test consistency verification)")
    
    # With precision gating
    print("\n[WITH Precision Gating]")
    executor_gated = LocalExecutor(enable_precision_gating=True)
    result_gated = executor_gated.execute(task)
    
    print(f"  State:        {result_gated.get('state_manifold', 'unknown')}")
    print(f"  Entropy:      {result_gated.get('entropy_metrics', {}).get('combined_entropy', 0):.4f}")
    print(f"  Confidence:   {result_gated.get('confidence', 0):.2f}")
    
    # Check for closure issues
    dualistic = result_gated.get('dualistic_verification', [])
    for prop in dualistic:
        if not prop.get('closure_valid'):
            print(f"  ⚠️  Closure Issue Detected: Gap = {prop.get('closure_gap', 0):.4f}")
    
    # Without precision gating
    print("\n[WITHOUT Precision Gating]")
    executor_plain = LocalExecutor(enable_precision_gating=False)
    result_plain = executor_plain.execute(task)
    
    print(f"  Output:       {str(result_plain.get('output', 'N/A'))[:60]}...")
    print(f"  Confidence:   {result_plain.get('confidence', 0):.2f}")
    print(f"  Note:         No entropy checking or closure verification")


def example_7_custom_configuration():
    """Example 7: Customizing precision gating behavior."""
    print("\n" + "="*70)
    print("Example 7: Custom Configuration")
    print("="*70)
    
    from h2q_project.local_executor import LocalExecutor
    
    # Create executor with precision gating
    executor = LocalExecutor(enable_precision_gating=True)
    
    if executor.precision_gated_executor:
        # Adjust precision threshold
        original_threshold = executor.precision_gated_executor._precision_threshold
        executor.precision_gated_executor._precision_threshold = 0.5
        
        print(f"Original precision threshold: {original_threshold:.2f}")
        print(f"New precision threshold:      {executor.precision_gated_executor._precision_threshold:.2f}")
        
        # Test with new threshold
        task = "A philosophical question"
        result = executor.execute(task)
        
        print(f"\nWith new threshold:")
        print(f"  Entropy: {result.get('entropy_metrics', {}).get('combined_entropy', 0):.4f}")
        print(f"  State:   {result.get('state_manifold', 'unknown')}")
        
        # Disable CoT for performance
        executor.precision_gated_executor.enable_cot = False
        print(f"\nChain-of-Thought disabled for performance")
        
        result2 = executor.execute("Another complex task")
        print(f"  Result with CoT disabled: {str(result2.get('output', 'N/A'))[:40]}...")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("DAS Meta-Theory Integration Examples")
    print("="*70)
    
    try:
        example_1_basic_usage()
        example_2_entropy_analysis()
        example_3_dualistic_verification()
        example_4_execution_tracing()
        example_5_statistics_and_monitoring()
        example_6_comparison_with_without_gating()
        example_7_custom_configuration()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
