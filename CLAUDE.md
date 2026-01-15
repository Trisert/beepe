# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Beepe** is a high-performance Byte Pair Encoding (BPE) tokenizer written in Rust with Python bindings via PyO3. It emphasizes memory efficiency (83% reduction vs tiktoken) and speed (10.5x faster encoding).

### Key Performance Characteristics
- **Encoding**: ~25.9M tokens/sec (10.5x faster than tiktoken)
- **Memory**: 18MB total (83% reduction from 106MB)
- **Training**: Entropy-weighted merge selection for better compression

## Workspace Structure

This is a Cargo workspace with the following crates:

```
beepe/
├── crates/
│   ├── core/          # Core BPE algorithms and data structures
│   ├── training/      # Training infrastructure (BpeTrainer)
│   ├── tokenizer/     # High-level tokenizer API with I/O
│   └── python/        # PyO3 Python bindings
└── cli/               # Command-line interface (beepe-cli)
```

### Crate Responsibilities

- **`beepe-core`**: Vocabulary storage, merge rules, encoding modes (byte/char level), special tokens
- **`beepe-training`**: `BpeTrainer` with parallel pair counting and entropy-weighted merge selection
- **`beepe-tokenizer`**: `Tokenizer` and `TokenizerBuilder` APIs, model I/O (HuggingFace/tiktoken formats)
- **`beepe-python`**: Python bindings via PyO3
- **`beepe-cli`**: `train`, `encode`, `decode`, `benchmark` commands

## Common Development Commands

### Rust Development

```bash
# Build all crates (development)
cargo build

# Build release mode (for benchmarks/CLI usage)
cargo build --release

# Run Rust tests
cargo test

# Run tests for specific crate
cargo test -p beepe-core
cargo test -p beepe-tokenizer

# Run benchmarks (criterion)
cargo bench
```

### Python Development

```bash
# Build and install Python package (development mode)
cd crates/python
maturin develop

# Using uv (recommended)
uv venv
source .venv/bin/activate
cd crates/python
maturin develop

# Run Python tests
pytest tests/

# Run specific test
pytest tests/test_basic.py::test_encode_decode_roundtrip

# Run with coverage
pytest --cov=beepe tests/
```

### CLI Usage

The CLI binary is built as `target/release/beepe`:

```bash
# Build CLI
cargo build --release -p beepe-cli

# Train a new tokenizer
./target/release/beepe train --input shakespeare.txt --vocab-size 30000

# Encode text
./target/release/beepe encode --model tokenizer.json --text "Hello world"

# Decode tokens
./target/release/beepe decode --model tokenizer.json --tokens "123 456 789"

# Run benchmarks
./target/release/beepe benchmark --model tokenizer.json --text shakespeare.txt
```

### Running Benchmarks

```bash
cd benchmarks
chmod +x run_benchmark.sh
./run_benchmark.sh

# Or manually
uv venv .venv
source .venv/bin/activate
uv pip install tiktoken sentencepiece numpy matplotlib
cd ..
cargo build --release -p beepe-cli
cd benchmarks
python benchmark.py --beepe-cli ../target/release/beepe
```

## Architecture Patterns

### Memory Optimization Patterns

**Arc sharing is critical for performance.** When creating multiple encoders from the same vocabulary:

```rust
use std::sync::Arc;
use beepe_core::ByteLevelEncoder;

// Get Arcs from existing encoder
let (vocab, vocab_r, merges) = encoder.get_arcs();

// Create new encoder with zero-copy sharing
let encoder2 = ByteLevelEncoder::with_arcs(vocab, vocab_r, merges);
```

### Layered Architecture

```
User Interface (Python/CLI)
    ↓
beepe-tokenizer (high-level API)
    ↓
beepe-core + beepe-training (algorithms)
    ↓
Data structures (AHashMap, compact_str, dary_heap)
```

### Key Data Structures

- **`AHashMap`**: O(1) hash map using aHash (faster than std::collections::HashMap)
- **`compact_str`**: Stack-allocated strings for small values (memory efficiency)
- **`dary_heap`**: D-ary heap for priority queue operations
- **`UnifiedVocab`**: Memory-efficient vocabulary storage

## Testing Strategy

### Test Files Location
- **Rust tests**: Co-located in `src/` files with `#[cfg(test)]` modules
- **Python tests**: `/home/nicola/beepe/tests/` directory
- **Fixtures**: `/home/nicola/beepe/tests/conftest.py` defines `sample_text`, `beepe_tokenizer`, `shakespeare_text`

### Property-Based Testing
- **Rust**: `proptest` for edge case testing (configured in workspace dependencies)
- **Python**: `hypothesis` (installed in `pyproject.toml`)

### Performance Baselines
Tests in `/home/nicola/beepe/tests/test_performance.py` enforce minimum performance:
- `encode_speed_tokens_per_ms`: 25000
- `decode_speed_tokens_per_ms`: 30000
- `max_slowdown_factor`: 2.0

## I/O and Serialization

### Supported Formats
- **HuggingFace**: `tokenizer.json` format (load/save in `beepe-tokenizer`)
- **Tiktoken**: Compatible with tiktoken models
- **Custom binary**: Optimized format for fast loading

### I/O Modules
- `crates/tokenizer/src/io/load.rs`: Model loading
- `crates/tokenizer/src/io/save.rs`: Model saving
- `crates/tokenizer/src/io/format.rs`: Format definitions

## Important Dependencies

### Performance-Critical Dependencies (workspace-level)
- `ahash`: Fast hashing for AHashMap
- `compact_str`: Stack-allocated small strings
- `dary_heap`: Efficient priority queues
- `rayon`: Parallel processing (used in `BpeTrainer`)

### Build Dependencies
- `maturin>=1.0`: PyO3 bridge for Python bindings
- PyO3 features: `extension-module` (enabled in `pyproject.toml`)

## Common Patterns

### Creating a Tokenizer (Python)
```python
import beepe

special_tokens = beepe.SpecialTokensConfig.new()
special_tokens.bos = "<bos>"
special_tokens.eos = "<eos>"

tokenizer = (
    beepe.Tokenizer.builder()
    .vocab_size(30000)
    .min_frequency(2)
    .with_special_tokens(special_tokens)
    .build()
)
tokenizer.train(text)
```

### Creating a Tokenizer (Rust)
```rust
use beepe_tokenizer::{Tokenizer, TokenizerBuilder, EncodingMode};

let tokenizer = TokenizerBuilder::new()
    .vocab_size(30000)
    .min_frequency(2)
    .encoding_mode(EncodingMode::ByteLevel)
    .build()?;
```

## File Processing Notes

- **`shakespeare.txt`**: Large test file (~5.4M characters) used for benchmarks
- Tests read first 10K chars via `shakespeare_text` fixture for speed
- Encoding operations process text in chunks to avoid memory issues

## Error Handling

- **Rust**: `thiserror` for detailed error types
- **Python**: Errors are exposed as Python exceptions via PyO3
- Error types defined in `crates/core/src/error.rs`

## Workspace Dependencies

All dependencies are defined in the root `Cargo.toml` workspace section. When adding new dependencies:
1. Add to `[workspace.dependencies]` if used across multiple crates
2. Use `crate-name = { workspace = true }` in individual crate `Cargo.toml` files
