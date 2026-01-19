#!/bin/bash
# H2Q-Evo Acceptance Test

echo "==================================================================="
echo "         H2Q-Evo Local Quantum AGI - Acceptance Test"
echo "==================================================================="
echo ""

PASS=0
FAIL=0

test_check() {
    if [ $1 -eq 0 ]; then
        echo "  PASS: $2"
        ((PASS++))
    else
        echo "  FAIL: $2"
        ((FAIL++))
    fi
}

# Test 1: Core Files
echo "Test 1: Core Files..."
test -f TERMINAL_AGI.py && test_check 0 "TERMINAL_AGI.py exists" || test_check 1 "TERMINAL_AGI.py missing"
test -f ENHANCED_LOCAL_AGI.py && test_check 0 "ENHANCED_LOCAL_AGI.py exists" || test_check 1 "ENHANCED_LOCAL_AGI.py missing"
test -f start_agi.sh && test_check 0 "start_agi.sh exists" || test_check 1 "start_agi.sh missing"

# Test 2: Documentation
echo -e "\nTest 2: Documentation..."
test -f LOCAL_AGI_GUIDE.md && test_check 0 "Guide exists" || test_check 1 "Guide missing"
test -f LOCAL_AGI_README.md && test_check 0 "README exists" || test_check 1 "README missing"
test -f DEMO_WALKTHROUGH.md && test_check 0 "Demo guide exists" || test_check 1 "Demo missing"

# Test 3: Python Syntax
echo -e "\nTest 3: Python Syntax..."
python3 -m py_compile TERMINAL_AGI.py 2>/dev/null && test_check 0 "TERMINAL_AGI syntax OK" || test_check 1 "TERMINAL_AGI syntax error"

# Test 4: Models
echo -e "\nTest 4: Model Files..."
MODEL_COUNT=$(find h2q_project -name "*.pth" -o -name "*.pt" 2>/dev/null | wc -l | tr -d ' ')
[ $MODEL_COUNT -ge 10 ] && test_check 0 "Found $MODEL_COUNT models (>=10)" || test_check 1 "Insufficient models: $MODEL_COUNT"

# Test 5: Functionality
echo -e "\nTest 5: Functionality..."
timeout 5 python3 -c "from TERMINAL_AGI import MathematicalProver; p=MathematicalProver(); r=p.prove_theorem('test'); assert 'proof_steps' in r" 2>/dev/null && test_check 0 "Math prover works" || test_check 1 "Math prover failed"

# Summary
echo -e "\n==================================================================="
echo "                        Test Summary"
echo "==================================================================="
echo ""
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo "  Total:  $((PASS + FAIL))"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "  SUCCESS: All tests passed!"
    exit 0
else
    echo "  WARNING: $FAIL tests failed"
    exit 1
fi
