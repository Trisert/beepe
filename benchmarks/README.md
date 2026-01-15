# Beepe Benchmarks

Benchmark comparison between **beepe**, **tiktoken**, and **sentencepiece**.

## Setup

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Rust and Cargo (for building beepe)

### Quick Start

The easiest way is to use the provided script:

```bash
cd benchmarks
chmod +x run_benchmark.sh
./run_benchmark.sh
```

This will:
1. Create a Python virtual environment with uv
2. Install all dependencies (tiktoken, sentencepiece, etc.)
3. Build beepe CLI in release mode
4. Run the benchmarks
5. Save results to `benchmark_results.json`

## Manual Setup

If you prefer to set up manually:

```bash
# Create Python environment
cd benchmarks
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install tiktoken sentencepiece numpy matplotlib

# Build beepe CLI
cd ..
cargo build --release -p beepe-cli
cd benchmarks

# Run benchmark
python benchmark.py --beepe-cli ../target/release/beepe
```

## Usage

```bash
python benchmark.py [OPTIONS]
```

### Options

- `--test-data PATH`: Path to test data file (generates random data if not provided)
- `--beepe-cli PATH`: Path to beepe CLI binary (default: `./target/release/beepe`)
- `--beepe-model PATH`: Path to trained beepe model (trains temp model if not provided)
- `--tiktoken-model NAME`: Tiktoken model name (default: `cl100k_base`)
- `--vocab-size N`: Vocabulary size for training (default: 30000)
- `--iterations N`: Number of benchmark iterations (default: 100)
- `--warmup N`: Number of warmup iterations (default: 10)
- `--output PATH`: Output JSON file for results (default: `benchmark_results.json`)
- `--skip-beepe`: Skip beepe benchmarking
- `--skip-tiktoken`: Skip tiktoken benchmarking
- `--skip-sentencepiece`: Skip sentencepiece benchmarking
- `--no-plot`: Don't display the plot interactively (useful for headless environments)

### Plotting

The benchmark automatically generates a visualization of results:

- A PNG file is saved to `<output>.png` (e.g., `benchmark_results.png`)
- If you're in a graphical environment, the plot will also be displayed interactively
- The plot shows:
  - **Encoding Time**: Bar chart comparing encoding speed (lower is better)
  - **Encoding Throughput**: Tokens processed per millisecond (higher is better)
  - **Decoding Time**: Bar chart comparing decoding speed (lower is better)

Example output:
```
Plot saved to benchmark_results.png
```

### Examples

```bash
# Run all benchmarks with default settings (shows plot at the end)
python benchmark.py

# Run with custom test data
python benchmark.py --test-data /path/to/text.txt

# Benchmark only beepe and tiktoken (skip sentencepiece training)
python benchmark.py --skip-sentencepiece

# Run with more iterations for more accurate results
python benchmark.py --iterations 1000 --warmup 100

# Use a smaller vocabulary for faster training
python benchmark.py --vocab-size 10000

# Run without interactive plot (headless mode)
python benchmark.py --no-plot
```

## Understanding the Results

The benchmark measures:
- **Encoding time**: Time to convert text to token IDs
- **Decoding time**: Time to convert token IDs back to text
- **Throughput**: Tokens processed per millisecond

Results are printed to console and saved to `benchmark_results.json` in the following format:

```json
{
  "beepe": {
    "encode (10000 chars)": [
      {"time_ms": 1.23, "tokens_per_ms": 1500}
    ],
    "decode (1500 tokens)": [
      {"time_ms": 0.45}
    ]
  },
  "tiktoken": {
    "encode (10000 chars)": [
      {"time_ms": 2.34, "tokens_per_ms": 800}
    ]
  }
}
```

## Notes

- **beepe** is benchmarked via CLI invocation, which includes subprocess overhead
- For pure Rust performance, you could benchmark beepe as a library directly
- **tiktoken** uses the `cl100k_base` encoding (GPT-4) by default
- **sentencepiece** requires training a model on the test data first
- Run benchmarks multiple times and average for most accurate results
- Close other applications while benchmarking for consistent results

## Expected Results

On modern hardware, you should see something like:

```
beepe:
  encode (10000 chars):     1.50 ± 0.10 ms (  2500 tokens/ms)
  decode (1500 tokens):     0.60 ± 0.05 ms

tiktoken:
  encode (10000 chars):     2.80 ± 0.20 ms (  1300 tokens/ms)
  decode (1500 tokens):     0.80 ± 0.06 ms

sentencepiece:
  encode (10000 chars):     3.50 ± 0.30 ms (  1000 tokens/ms)
  decode (1500 tokens):     1.20 ± 0.10 ms
```

Your actual results will vary based on hardware, text complexity, and vocabulary size.
