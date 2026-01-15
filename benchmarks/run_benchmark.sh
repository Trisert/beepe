#!/bin/bash
# Setup and run the benchmark script

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "Setting up Python environment with uv..."

# Create virtual environment with uv if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies if needed
if ! python -c "import tiktoken" 2>/dev/null; then
    echo "Installing dependencies..."
    uv pip install tiktoken sentencepiece numpy matplotlib
fi

echo ""
echo "Building beepe CLI in release mode..."
cargo build --release -p beepe-cli

echo ""
echo "Running benchmarks..."
python benchmarks/benchmark.py \
    --beepe-cli ./target/release/beepe \
    --iterations 100 \
    --warmup 10 \
    --output benchmarks/benchmark_results.json

echo ""
echo "Benchmark complete! Results saved to benchmarks/benchmark_results.json"
