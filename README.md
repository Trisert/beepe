# Beepe

A high-performance, memory-efficient Byte Pair Encoding (BPE) tokenizer written in Rust with Python bindings.

## Features

- **Ultra-fast encoding**: ~25.9M tokens/sec (10.5x faster than tiktoken)
- **Memory efficient**: 83% memory reduction through Arc sharing and optimized data structures
- **Entropy-weighted training**: Better compression through intelligent merge selection
- **Multiple encoding modes**: Byte-level and character-level with optimized encoding
- **Python bindings**: Easy-to-use Python API via PyO3
- **CLI tool**: Command-line interface for training, encoding, decoding, and benchmarking
- **LLM-ready**: Special tokens support, chat templates, and role tokens

## Performance

### Encoding Speed (tokens/second)

| Tokenizer  | Speed   | vs Beepe |
|------------|---------|----------|
| Beepe      | 25.9M   | 1.0x     |
| Tiktoken   | 2.5M    | 10.5x slower |
| SentencePiece | ~2M  | 13x slower |

### Memory Usage

| Metric     | Beepe | vs Tiktoken | vs SentencePiece |
|------------|-------|-------------|------------------|
| Total memory | 18 MB | 106 MB | ~80 MB |
| Encoding structure | 7.5 KB | 8.6 MB | ~2 MB |
| Training memory | 1.5 MB | 27 MB | ~25 MB |

## Installation

### From PyPI (Python)

```bash
pip install beepe
```

### From Source

```bash
# Clone the repository
git clone https://github.com/Trisert/beepe.git
cd beepe

# Build and install Python package
cd crates/python
maturin develop --release

# Or build the CLI
cargo build --release
```

## Quick Start

### Python

```python
import beepe

# Create a tokenizer
tokenizer = (
    beepe.Tokenizer.builder()
    .vocab_size(30000)
    .min_frequency(2)
    .build()
)

# Train on your text
tokenizer.train("Hello, world! This is a test.")

# Encode text to tokens
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [15496, 11, 1917, 0]

# Decode tokens back to text
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"
```

### Command Line

```bash
# Train a new tokenizer
beepe train --input text.txt --vocab-size 30000 --output tokenizer.json

# Encode text
beepe encode --model tokenizer.json --text "Hello, world!"

# Decode tokens
beepe decode --model tokenizer.json --tokens "15496 11 1917 0"

# Benchmark performance
beepe benchmark --model tokenizer.json --text text.txt
```

### Rust

```rust
use beepe_tokenizer::{Tokenizer, TokenizerBuilder, EncodingMode};

let mut tokenizer = TokenizerBuilder::new()
    .vocab_size(30000)
    .min_frequency(2)
    .encoding_mode(EncodingMode::ByteLevel)
    .build()?;

tokenizer.train("Hello, world! This is a test.")?;

let tokens = tokenizer.encode("Hello, world!")?;
let text = tokenizer.decode(&tokens)?;
```

## Advanced Usage

### Special Tokens

```python
import beepe

special_tokens = beepe.SpecialTokensConfig.new()
special_tokens.bos = "<bos>"
special_tokens.eos = "<eos>"
special_tokens.pad = "<pad>"
special_tokens.unk = "<unk>"

tokenizer = (
    beepe.Tokenizer.builder()
    .with_special_tokens(special_tokens)
    .build()
)
```

### Using Arc Sharing (Zero-Copy)

```python
import beepe

# Create primary tokenizer
tokenizer1 = beepe.Tokenizer.builder().build()

# Share vocabulary (zero-copy, no memory duplication)
tokenizer2 = tokenizer1.shallow_copy()
```

### Loading and Saving Models

```python
import beepe

# Save trained tokenizer
tokenizer = beepe.Tokenizer.builder().build()
tokenizer.train(text)
tokenizer.save("tokenizer.json")

# Load from file
tokenizer = beepe.Tokenizer.load("tokenizer.json")
```

## Development

### Building

```bash
# Build all crates
cargo build

# Build in release mode
cargo build --release

# Build CLI only
cargo build --release -p beepe-cli
```

### Testing

```bash
# Run Rust tests
cargo test

# Run Python tests
cd crates/python
maturin develop
pytest tests/

# Run benchmarks
cargo bench
```

### Python Development

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install development dependencies
cd crates/python
uv pip install -e ".[test]"

# Run tests
pytest tests/
```

## Benchmarks

Run the comprehensive benchmark suite to compare with tiktoken and sentencepiece:

```bash
cd benchmarks
chmod +x run_benchmark.sh
./run_benchmark.sh
```

Or manually:

```bash
# Set up Python environment
cd benchmarks
uv venv .venv
source .venv/bin/activate
uv pip install tiktoken sentencepiece numpy matplotlib

# Build beepe CLI
cd ..
cargo build --release -p beepe-cli
cd benchmarks

# Run benchmark
python benchmark.py --beepe-cli ../target/release/beepe
```

## Architecture

Beepe is organized as a Cargo workspace with the following crates:

- **`beepe-core`**: Core BPE algorithm implementation with vocabulary management and encoding modes
- **`beepe-training`**: Training infrastructure with parallel pair counting and entropy-weighted merge selection
- **`beepe-tokenizer`**: High-level tokenizer API with model I/O (HuggingFace/tiktoken formats)
- **`beepe-python`**: Python bindings via PyO3
- **`beepe-cli`**: Command-line interface

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `cargo test` and `pytest tests/`
5. Run benchmarks to verify performance: `cd benchmarks && ./run_benchmark.sh`
6. Submit a pull request

## License

[Specify your license here]

## Acknowledgments

- Inspired by [tiktoken](https://github.com/openai/tiktoken) and [sentencepiece](https://github.com/google/sentencepiece)
- Uses [PyO3](https://github.com/PyO3/pyo3) for Python bindings
- Built with [Rust](https://www.rust-lang.org/)
