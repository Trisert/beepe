#!/usr/bin/env python3
"""
Comprehensive benchmark for beepe tokenizer.

This script benchmarks:
1. Training performance (time, throughput, memory)
2. Encoding speed (time, throughput, tokens/sec)
3. Decoding speed (time, throughput)
4. Memory usage (peak, resident)
5. Compression ratios
6. Comparison with tiktoken
"""

import argparse
import gc
import json
import os
import resource
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

import numpy as np
import psutil

# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import tiktoken
except ImportError:
    tiktoken = None
    print("Warning: tiktoken not installed, skipping comparison")

# Add parent directory to path for imports (not needed with proper install)
# sys.path.insert(0, str(Path(__file__).parent.parent / "crates" / "python" / "python"))
try:
    import beepe
except ImportError:
    beepe = None
    print("Warning: beepe not installed. Install with: maturin develop --release")
    print("Or from the project root: uv pip install -e crates/python")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size
        "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size
        "peak_mb": mem_info.rss / 1024 / 1024,  # Peak (approximated by RSS for now)
    }


def format_bytes(bytes_val: int) -> str:
    """Format bytes in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def format_number(n: int) -> str:
    """Format large numbers with commas."""
    return f"{n:,}"


class BenchmarkMetrics:
    """Store comprehensive benchmark metrics."""

    def __init__(self, name: str):
        self.name = name
        self.metrics = {}

    def add_training_metrics(self, train_time_sec: float, text_size: int,
                            vocab_size: int, memory_before: Dict, memory_after: Dict):
        """Add training performance metrics."""
        self.metrics['training'] = {
            'time_sec': train_time_sec,
            'time_ms': train_time_sec * 1000,
            'text_size': text_size,
            'text_size_mb': text_size / 1024 / 1024,
            'vocab_size': vocab_size,
            'throughput_chars_sec': text_size / train_time_sec,
            'throughput_mb_sec': (text_size / 1024 / 1024) / train_time_sec,
            'memory_before_mb': memory_before['rss_mb'],
            'memory_after_mb': memory_after['rss_mb'],
            'memory_used_mb': memory_after['rss_mb'] - memory_before['rss_mb'],
            'peak_memory_mb': memory_after['rss_mb'],  # Approximate
        }

    def add_encode_metrics(self, encode_time_ms: float, text_size: int,
                          token_count: int, iterations: int = 1):
        """Add encoding performance metrics."""
        time_per_iter = encode_time_ms / iterations
        self.metrics['encode'] = {
            'time_ms': encode_time_ms,
            'time_per_iter_ms': time_per_iter,
            'text_size': text_size,
            'token_count': token_count,
            'iterations': iterations,
            'throughput_chars_sec': (text_size * iterations) / (encode_time_ms / 1000),
            'throughput_tokens_sec': (token_count * iterations) / (encode_time_ms / 1000),
            'compression_ratio': text_size / token_count,
            'chars_per_token': text_size / token_count,
        }

    def add_decode_metrics(self, decode_time_ms: float, token_count: int,
                          iterations: int = 1):
        """Add decoding performance metrics."""
        time_per_iter = decode_time_ms / iterations
        self.metrics['decode'] = {
            'time_ms': decode_time_ms,
            'time_per_iter_ms': time_per_iter,
            'token_count': token_count,
            'iterations': iterations,
            'throughput_tokens_sec': (token_count * iterations) / (decode_time_ms / 1000),
        }

    def print_summary(self):
        """Print a formatted summary of all metrics."""
        print(f"\n{'='*80}")
        print(f"{self.name.upper()}")
        print(f"{'='*80}")

        # Training metrics
        if 'training' in self.metrics:
            t = self.metrics['training']
            print(f"\nTraining:")
            print(f"  Time: {t['time_sec']:.2f}s")
            print(f"  Text size: {format_number(t['text_size'])} chars ({t['text_size_mb']:.2f} MB)")
            print(f"  Vocab size: {format_number(t['vocab_size'])}")
            print(f"  Throughput: {format_number(int(t['throughput_chars_sec']))} chars/sec")
            print(f"             {t['throughput_mb_sec']:.2f} MB/sec")
            print(f"  Memory usage: {t['memory_used_mb']:.2f} MB")
            print(f"  Peak memory: {t['peak_memory_mb']:.2f} MB")

        # Encoding metrics
        if 'encode' in self.metrics:
            e = self.metrics['encode']
            print(f"\nEncoding:")
            print(f"  Time: {e['time_ms']:.2f}ms ({e['iterations']} iterations)")
            print(f"  Text size: {format_number(e['text_size'])} chars")
            print(f"  Tokens: {format_number(e['token_count'])}")
            print(f"  Throughput: {format_number(int(e['throughput_chars_sec']))} chars/sec")
            print(f"              {format_number(int(e['throughput_tokens_sec']))} tokens/sec")
            print(f"  Compression: {e['compression_ratio']:.2f} chars/token")

        # Decoding metrics
        if 'decode' in self.metrics:
            d = self.metrics['decode']
            print(f"\nDecoding:")
            print(f"  Time: {d['time_ms']:.2f}ms ({d['iterations']} iterations)")
            print(f"  Tokens: {format_number(d['token_count'])}")
            print(f"  Throughput: {format_number(int(d['throughput_tokens_sec']))} tokens/sec")


def benchmark_beepe_training(text_path: Path, output_path: Path,
                              vocab_size: int = 30000) -> BenchmarkMetrics:
    """Benchmark beepe training performance."""
    print(f"\n{'='*80}")
    print("BENCHMARKING BEEPE TRAINING")
    print(f"{'='*80}")

    import beepe

    # Read text
    with open(text_path, 'r') as f:
        text = f.read()

    print(f"Training data: {format_number(len(text))} chars ({len(text)/1024/1024:.2f} MB)")

    # Get initial memory
    gc.collect()
    mem_before = get_memory_usage()

    # Train tokenizer
    start = time.time()
    tokenizer = beepe.Tokenizer.builder() \
        .vocab_size(vocab_size) \
        .encoding_mode(beepe.EncodingMode.byte_level()) \
        .build()
    tokenizer.train(text)
    train_time = time.time() - start

    mem_after = get_memory_usage()

    # Save model
    tokenizer.save(str(output_path))

    print(f"Training completed in {train_time:.2f}s")
    print(f"Final vocab size: {tokenizer.vocab_size()}")

    # Create metrics
    metrics = BenchmarkMetrics("beepe")
    metrics.add_training_metrics(train_time, len(text), tokenizer.vocab_size(),
                                mem_before, mem_after)

    return metrics


def benchmark_encoding(tokenizer, text: str, name: str,
                      iterations: int = 10) -> Dict[str, float]:
    """Benchmark encoding performance."""
    print(f"\nBenchmarking {name} encoding...")

    # Check if this is a beepe tokenizer (has vocab_size method)
    is_beepe = hasattr(tokenizer, 'vocab_size')

    # Warm up
    for _ in range(3):
        if is_beepe:
            result = tokenizer.encode(text[:10000], False)
        else:
            # tiktoken
            result = tokenizer.encode(text[:10000])

    gc.collect()
    mem_before = get_memory_usage()

    # Benchmark
    start = time.time()
    tokens = None
    for _ in range(iterations):
        if is_beepe:
            tokens = tokenizer.encode(text, False)
        else:
            # tiktoken
            tokens = tokenizer.encode(text)
    elapsed_ms = (time.time() - start) * 1000

    mem_after = get_memory_usage()

    token_count = len(tokens) if tokens else 0

    print(f"  {iterations} iterations in {elapsed_ms:.2f}ms")
    print(f"  Tokens: {format_number(token_count)}")
    print(f"  Throughput: {format_number(int(len(text) / (elapsed_ms / 1000)))} chars/sec")
    print(f"              {format_number(int(token_count / (elapsed_ms / 1000)))} tokens/sec")
    print(f"  Memory: {mem_after['rss_mb'] - mem_before['rss_mb']:.2f} MB")

    return {
        'time_ms': elapsed_ms,
        'tokens': token_count,
        'memory_mb': mem_after['rss_mb'] - mem_before['rss_mb'],
    }


def benchmark_decoding(tokenizer, tokens: List[int], name: str,
                      iterations: int = 10) -> Dict[str, float]:
    """Benchmark decoding performance."""
    print(f"\nBenchmarking {name} decoding...")

    # Check if this is a beepe tokenizer (has vocab_size method)
    is_beepe = hasattr(tokenizer, 'vocab_size')

    # Warm up
    for _ in range(3):
        if is_beepe:
            tokenizer.decode(tokens[:1000], False)
        else:
            # tiktoken
            tokenizer.decode(tokens[:1000])

    gc.collect()
    mem_before = get_memory_usage()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        if is_beepe:
            decoded = tokenizer.decode(tokens, False)
        else:
            # tiktoken
            decoded = tokenizer.decode(tokens)
    elapsed_ms = (time.time() - start) * 1000

    mem_after = get_memory_usage()

    print(f"  {iterations} iterations in {elapsed_ms:.2f}ms")
    print(f"  Throughput: {format_number(int(len(tokens) / (elapsed_ms / 1000)))} tokens/sec")
    print(f"  Memory: {mem_after['rss_mb'] - mem_before['rss_mb']:.2f} MB")

    return {
        'time_ms': elapsed_ms,
        'memory_mb': mem_after['rss_mb'] - mem_before['rss_mb'],
    }


def run_comprehensive_benchmark(text_path: Path, model_path: Path,
                                vocab_size: int = 30000,
                                iterations: int = 10) -> Dict[str, Any]:
    """Run comprehensive benchmark comparing beepe and tiktoken."""

    print(f"\n{'='*80}")
    print("COMPREHENSIVE TOKENIZER BENCHMARK")
    print(f"{'='*80}")
    print(f"Text: {text_path}")
    print(f"Model output: {model_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Iterations: {iterations}")

    # Read test data
    with open(text_path, 'r') as f:
        text = f.read()

    print(f"\nTest data: {format_number(len(text))} chars ({len(text)/1024/1024:.2f} MB)")

    results = {}

    # Benchmark beepe training
    if beepe:
        beepe_train_metrics = benchmark_beepe_training(text_path, model_path, vocab_size)

        # Load the trained model for encode/decode benchmarks
        tokenizer = beepe.Tokenizer.load(str(model_path))

        # Benchmark encoding
        enc_metrics = benchmark_encoding(tokenizer, text, "beepe", iterations)
        beepe_train_metrics.add_encode_metrics(
            enc_metrics['time_ms'], len(text), enc_metrics['tokens'], iterations
        )

        # Benchmark decoding
        beepe_tokens = tokenizer.encode(text, False)
        dec_metrics = benchmark_decoding(tokenizer, beepe_tokens, "beepe", iterations)
        beepe_train_metrics.add_decode_metrics(
            dec_metrics['time_ms'], len(beepe_tokens), iterations
        )

        results['beepe'] = beepe_train_metrics

    # Benchmark tiktoken (for comparison)
    if tiktoken:
        print(f"\n{'='*80}")
        print("BENCHMARKING TIKTOKEN (for comparison)")
        print(f"{'='*80}")

        tiktoken_metrics = BenchmarkMetrics("tiktoken (cl100k_base)")
        encoder = tiktoken.encoding_for_model("gpt-4")

        # Benchmark encoding
        enc_metrics = benchmark_encoding(encoder, text, "tiktoken", iterations)
        tiktoken_metrics.add_encode_metrics(
            enc_metrics['time_ms'], len(text), enc_metrics['tokens'], iterations
        )

        # Benchmark decoding
        tiktoken_tokens = encoder.encode(text)
        dec_metrics = benchmark_decoding(encoder, tiktoken_tokens, "tiktoken", iterations)
        tiktoken_metrics.add_decode_metrics(
            dec_metrics['time_ms'], len(tiktoken_tokens), iterations
        )

        results['tiktoken'] = tiktoken_metrics

    # Print summary comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    if 'beepe' in results and 'tiktoken' in results:
        beepe_enc = results['beepe'].metrics['encode']
        tiktoken_enc = results['tiktoken'].metrics['encode']

        beepe_dec = results['beepe'].metrics['decode']
        tiktoken_dec = results['tiktoken'].metrics['decode']

        print(f"\nEncode speed:")
        encode_speedup = tiktoken_enc['time_ms'] / beepe_enc['time_ms']
        print(f"  beepe: {beepe_enc['time_ms']:.2f}ms")
        print(f"  tiktoken: {tiktoken_enc['time_ms']:.2f}ms")
        print(f"  Speedup: beepe is {encode_speedup:.2f}x {'faster' if encode_speedup > 1 else 'slower'}")

        print(f"\nDecode speed:")
        decode_speedup = tiktoken_dec['time_ms'] / beepe_dec['time_ms']
        print(f"  beepe: {beepe_dec['time_ms']:.2f}ms")
        print(f"  tiktoken: {tiktoken_dec['time_ms']:.2f}ms")
        print(f"  Speedup: beepe is {decode_speedup:.2f}x {'faster' if decode_speedup > 1 else 'slower'}")

        print(f"\nCompression:")
        print(f"  beepe: {beepe_enc['compression_ratio']:.2f} chars/token")
        print(f"  tiktoken: {tiktoken_enc['compression_ratio']:.2f} chars/token")

    # Print individual summaries
    for name, metrics in results.items():
        metrics.print_summary()

    return results


def create_visualization(results: Dict[str, Any], output_path: Path):
    """Create comprehensive visualization of benchmark results."""

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Training metrics (bar chart)
    if 'beepe' in results and 'training' in results['beepe'].metrics:
        ax1 = fig.add_subplot(gs[0, 0])
        t = results['beepe'].metrics['training']

        metrics = ['Time (s)', 'Throughput\n(MB/s)', 'Memory (MB)']
        values = [
            t['time_sec'],
            t['throughput_mb_sec'],
            t['memory_used_mb']
        ]

        bars = ax1.bar(metrics, values, color=['#1f77b4', '#2ca02c', '#d62728'])
        ax1.set_ylabel('Value')
        ax1.set_title('Beepe Training Metrics')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    # 2. Encode throughput comparison
    if 'beepe' in results and 'tiktoken' in results:
        ax2 = fig.add_subplot(gs[0, 1])
        beepe_tps = results['beepe'].metrics['encode']['throughput_tokens_sec'] / 1000
        tiktoken_tps = results['tiktoken'].metrics['encode']['throughput_tokens_sec'] / 1000

        x = ['beepe', 'tiktoken']
        y = [beepe_tps, tiktoken_tps]

        bars = ax2.bar(x, y, color=['#1f77b4', '#ff7f0e'])
        ax2.set_ylabel('Throughput (K tokens/sec)')
        ax2.set_title('Encoding Throughput Comparison')

        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}K', ha='center', va='bottom', fontsize=10)

    # 3. Decode throughput comparison
    if 'beepe' in results and 'tiktoken' in results:
        ax3 = fig.add_subplot(gs[0, 2])
        beepe_tps = results['beepe'].metrics['decode']['throughput_tokens_sec'] / 1000
        tiktoken_tps = results['tiktoken'].metrics['decode']['throughput_tokens_sec'] / 1000

        x = ['beepe', 'tiktoken']
        y = [beepe_tps, tiktoken_tps]

        bars = ax3.bar(x, y, color=['#1f77b4', '#ff7f0e'])
        ax3.set_ylabel('Throughput (K tokens/sec)')
        ax3.set_title('Decoding Throughput Comparison')

        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}K', ha='center', va='bottom', fontsize=10)

    # 4. Encode time comparison (lower is better)
    if 'beepe' in results and 'tiktoken' in results:
        ax4 = fig.add_subplot(gs[1, 0])
        beepe_time = results['beepe'].metrics['encode']['time_ms']
        tiktoken_time = results['tiktoken'].metrics['encode']['time_ms']

        x = ['beepe', 'tiktoken']
        y = [beepe_time, tiktoken_time]

        bars = ax4.bar(x, y, color=['#1f77b4', '#ff7f0e'])
        ax4.set_ylabel('Time (ms)')
        ax4.set_title('Encoding Time (Lower is Better)')

        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}ms', ha='center', va='bottom', fontsize=10)

    # 5. Decode time comparison (lower is better)
    if 'beepe' in results and 'tiktoken' in results:
        ax5 = fig.add_subplot(gs[1, 1])
        beepe_time = results['beepe'].metrics['decode']['time_ms']
        tiktoken_time = results['tiktoken'].metrics['decode']['time_ms']

        x = ['beepe', 'tiktoken']
        y = [beepe_time, tiktoken_time]

        bars = ax5.bar(x, y, color=['#1f77b4', '#ff7f0e'])
        ax5.set_ylabel('Time (ms)')
        ax5.set_title('Decoding Time (Lower is Better)')

        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.0f}ms', ha='center', va='bottom', fontsize=10)

    # 6. Compression comparison
    if 'beepe' in results and 'tiktoken' in results:
        ax6 = fig.add_subplot(gs[1, 2])
        beepe_comp = results['beepe'].metrics['encode']['compression_ratio']
        tiktoken_comp = results['tiktoken'].metrics['encode']['compression_ratio']

        x = ['beepe', 'tiktoken']
        y = [beepe_comp, tiktoken_comp]

        bars = ax6.bar(x, y, color=['#1f77b4', '#ff7f0e'])
        ax6.set_ylabel('Chars per Token')
        ax6.set_title('Compression Ratio (Higher is Better)')

        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Comprehensive Tokenizer Benchmark Results', fontsize=16, y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_path}")


def save_results_json(results: Dict[str, Any], output_path: Path):
    """Save results to JSON file."""
    output = {}
    for name, metrics in results.items():
        output[name] = metrics.metrics

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive tokenizer benchmark')
    parser.add_argument('--text', type=Path, default=Path('shakespeare.txt'),
                       help='Path to training/test text file')
    parser.add_argument('--model', type=Path, default=Path('/tmp/shakespeare_beepe_benchmark'),
                       help='Path to save/load model')
    parser.add_argument('--vocab-size', type=int, default=30000,
                       help='Target vocabulary size')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of benchmark iterations')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training, use existing model')

    args = parser.parse_args()

    if not args.text.exists():
        print(f"Error: Text file not found: {args.text}")
        sys.exit(1)

    # Check if beepe is installed
    if not beepe:
        print("Error: beepe not installed. Run: cd crates/python && maturin develop --release")
        sys.exit(1)

    # Run benchmarks
    if not args.skip_training:
        results = run_comprehensive_benchmark(
            args.text,
            args.model,
            args.vocab_size,
            args.iterations
        )
    else:
        # Use existing model, skip training
        print(f"Skipping training, using existing model from {args.model}")

        with open(args.text, 'r') as f:
            text = f.read()

        print(f"\nTest data: {format_number(len(text))} chars")

        results = {}
        tokenizer = beepe.Tokenizer.load(str(args.model))

        # Benchmark encoding
        enc_metrics = benchmark_encoding(tokenizer, text, "beepe", args.iterations)
        beepe_metrics = BenchmarkMetrics("beepe")
        beepe_metrics.add_encode_metrics(
            enc_metrics['time_ms'], len(text), enc_metrics['tokens'], args.iterations
        )

        # Benchmark decoding
        beepe_tokens = tokenizer.encode(text, False)
        dec_metrics = benchmark_decoding(tokenizer, beepe_tokens, "beepe", args.iterations)
        beepe_metrics.add_decode_metrics(
            dec_metrics['time_ms'], len(beepe_tokens), args.iterations
        )

        results['beepe'] = beepe_metrics

        # Compare with tiktoken if available
        if tiktoken:
            print(f"\n{'='*80}")
            print("BENCHMARKING TIKTOKEN (for comparison)")
            print(f"{'='*80}")

            tiktoken_metrics = BenchmarkMetrics("tiktoken (cl100k_base)")
            encoder = tiktoken.encoding_for_model("gpt-4")

            enc_metrics = benchmark_encoding(encoder, text, "tiktoken", args.iterations)
            tiktoken_metrics.add_encode_metrics(
                enc_metrics['time_ms'], len(text), enc_metrics['tokens'], args.iterations
            )

            tiktoken_tokens = encoder.encode(text)
            dec_metrics = benchmark_decoding(encoder, tiktoken_tokens, "tiktoken", args.iterations)
            tiktoken_metrics.add_decode_metrics(
                dec_metrics['time_ms'], len(tiktoken_tokens), args.iterations
            )

            results['tiktoken'] = tiktoken_metrics

        # Print summaries
        for name, metrics in results.items():
            metrics.print_summary()

        # Print comparison
        if 'beepe' in results and 'tiktoken' in results:
            print(f"\n{'='*80}")
            print("COMPARISON SUMMARY")
            print(f"{'='*80}")

            beepe_enc = results['beepe'].metrics['encode']
            tiktoken_enc = results['tiktoken'].metrics['encode']

            beepe_dec = results['beepe'].metrics['decode']
            tiktoken_dec = results['tiktoken'].metrics['decode']

            print(f"\nEncode speed:")
            encode_speedup = tiktoken_enc['time_ms'] / beepe_enc['time_ms']
            print(f"  beepe: {beepe_enc['time_ms']:.2f}ms")
            print(f"  tiktoken: {tiktoken_enc['time_ms']:.2f}ms")
            print(f"  Speedup: beepe is {encode_speedup:.2f}x {'faster' if encode_speedup > 1 else 'slower'}")

            print(f"\nDecode speed:")
            decode_speedup = tiktoken_dec['time_ms'] / beepe_dec['time_ms']
            print(f"  beepe: {beepe_dec['time_ms']:.2f}ms")
            print(f"  tiktoken: {tiktoken_dec['time_ms']:.2f}ms")
            print(f"  Speedup: beepe is {decode_speedup:.2f}x {'faster' if decode_speedup > 1 else 'slower'}")

    # Save results
    results_dir = Path("benchmarks/results")
    results_dir.mkdir(exist_ok=True)

    json_path = results_dir / "comprehensive_benchmark_results.json"
    save_results_json(results, json_path)

    plot_path = results_dir / "comprehensive_benchmark_results.png"
    create_visualization(results, plot_path)


if __name__ == '__main__':
    main()
