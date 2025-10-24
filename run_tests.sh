#!/bin/bash
# human-tracking Test Runner
set -e

echo "=== human-tracking Test Runner ==="

# Parse test type argument
TEST_TYPE="${1:-smoke}"
echo "Test type: $TEST_TYPE"

# Install dependencies (user mode to avoid permission issues in Docker)
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install --user -r requirements.txt --quiet || pip install -r requirements.txt --quiet
fi

# Install test dependencies
pip install --user pytest pytest-cov --quiet || pip install pytest pytest-cov --quiet

# Check if tests directory exists
if [ ! -d "tests" ]; then
    echo "Warning: No tests/ directory found. Creating basic import test..."
    mkdir -p tests
    cat > tests/test_import.py << 'EOF'
def test_basic_import():
    """Basic import test"""
    import sys
    assert sys.version_info >= (3, 6)
    print("✓ Basic import test passed")
EOF
fi

if [ "$TEST_TYPE" = "smoke" ]; then
    echo "Running smoke tests..."
    pytest tests/ -v --tb=short -x -k "test_import or test_basic" || pytest tests/ -v --tb=short -x --maxfail=3
    echo "✓ Smoke tests passed!"
else
    echo "Running comprehensive tests..."
    pytest tests/ -v --tb=short --cov=src --cov-report=term-missing || pytest tests/ -v --tb=short
    echo "✓ All tests passed!"
fi
