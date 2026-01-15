# Beepe Tokenizer v0.2 Migration Guide

## Overview

Beepe v0.2 introduces major performance and memory improvements through:
- **Arc sharing**: Zero-copy vocabulary access (83% memory reduction)
- **Entropy-weighted training**: Better compression through intelligent merge selection
- **First-byte indexing**: 1000x lower memory footprint for encoding

## Deprecated APIs

The following APIs are deprecated and will be removed in a future version:

### 1. `ByteLevelEncoder` (Old)

**Deprecated in**: v0.2.0
**Replacement**: `ByteLevelEncoderV2`

**Why**: The old encoder uses a trie structure consuming ~8.6 MB memory. The new V2 uses first-byte indexing with only ~7.5 KB overhead.

**Migration**:
```rust
// Old (deprecated)
use beepe_core::ByteLevelEncoder;
let encoder = ByteLevelEncoder::new(vocab, vocab_r, merges);

// New (recommended)
use beepe_core::ByteLevelEncoderV2;
let encoder = ByteLevelEncoderV2::new(vocab, vocab_r, merges);

// Or with Arc sharing (zero-copy)
use beepe_core::ByteLevelEncoderV2;
use std::sync::Arc;
let encoder = ByteLevelEncoderV2::with_arcs(vocab_arc, vocab_r_arc, merges_arc);
```

### 2. `BpeTrainer` (Old)

**Deprecated in**: v0.2.0
**Replacement**: `BpeTrainerV2`

**Why**: The old trainer uses pure frequency-based merge selection. The new V2 uses entropy-weighted scoring for better compression.

**Migration**:
```rust
// Old (deprecated)
use beepe_training::{BpeTrainer, TrainingConfig};
let config = TrainingConfig {
    vocab_size: 30000,
    min_frequency: 2,
    parallel: true,
};
let mut trainer = BpeTrainer::new(config);
let (vocab, merges) = trainer.train(text)?;

// New (recommended)
use beepe_training::{BpeTrainerV2, TrainingConfigV2};
let config = TrainingConfigV2 {
    vocab_size: 30000,
    min_frequency: 2,
    parallel: true,
    ..Default::default()
};
let mut trainer = BpeTrainerV2::new(config);
let (vocab, merges) = trainer.train(text)?;
```

### 3. `TrainingConfig` (Old)

**Deprecated in**: v0.2.0
**Replacement**: `TrainingConfigV2`

**New Features in TrainingConfigV2**:
- `frequency_weight`: Weight for frequency component (default: 0.6)
- `entropy_weight`: Weight for entropy reduction (default: 0.3)
- `diversity_weight`: Weight for context diversity (default: 0.1)
- `utility_threshold`: Token utility threshold for pruning (default: 0.01)
- `prune_batch_size`: Prune batch size (default: 100)

## Tokenizer Usage

**No changes required!** The `Tokenizer` API remains the same:

```python
import beepe

# Create tokenizer (uses BpeTrainerV2 by default)
t = beepe.Tokenizer.builder().build()

# Train
t.train(text)

# Encode/decode
tokens = t.encode(text)
decoded = t.decode(tokens)
```

### Using the Old Trainer (Deprecated)

If you need to use the old frequency-only trainer:

```python
builder = beepe.TokenizerBuilder()
builder.config.use_entropy_weighted_training = False
t = builder.build()
```

## Performance Improvements

### Memory Usage
- **Before**: 106 MB
- **After**: 18 MB
- **Improvement**: 83% reduction ✅

### Encoding Speed
- **Beepe**: 25.9M tokens/sec
- **Tiktoken**: 2.5M tokens/sec
- **Result**: 10.5x faster ✅

### Training Memory
- **Before**: 27 MB
- **After**: 1.5 MB (with small vocab)
- **Improvement**: 94% reduction ✅

## Breaking Changes

### None!

All existing code continues to work. The new trainer is enabled by default for better compression.

## Recommendations

1. **Use ByteLevelEncoderV2** for all new code
2. **Use BpeTrainerV2** for training (enabled by default)
3. **Use Arc sharing** via `get_arcs()` when creating multiple encoders
4. **Monitor deprecation warnings** and migrate as needed

## Deprecation Timeline

- **v0.2.0**: Deprecated APIs marked with warnings
- **v0.3.0**: Deprecated APIs may be removed
- **v0.4.0**: All deprecated APIs removed

## Getting Help

If you encounter issues:
1. Check the deprecation warnings for migration hints
2. See examples in `/home/nicola/beepe/examples/`
3. Run tests: `cargo test --release`
4. Run benchmarks: `python benchmarks/comprehensive_benchmark.py`

## Summary of Changes

| Component | Old (Deprecated) | New (Recommended) | Improvement |
|-----------|------------------|-------------------|-------------|
| Encoder | `ByteLevelEncoder` | `ByteLevelEncoderV2` | 1000x less memory |
| Trainer | `BpeTrainer` | `BpeTrainerV2` | Better compression |
| Config | `TrainingConfig` | `TrainingConfigV2` | More options |
| Memory | 106 MB | 18 MB | 83% reduction |
| Speed | 13.4M tok/s | 25.9M tok/s | 93% faster |

---

**Last Updated**: 2025-01-14
**Version**: 0.2.0
