#!/usr/bin/env python3
"""
Benchmark beepe against tiktoken and sentencepiece.

This script compares the performance of three BPE tokenizer implementations:
- beepe: Our Rust-based tokenizer (native Python bindings)
- tiktoken: OpenAI's tokenizer (used in GPT-3/4)
- sentencepiece: Google's tokenizer (used in T5, BERT, etc.)
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

# Use non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class BenchmarkResults:
    """Store and display benchmark results."""

    def __init__(self):
        self.results = {}

    def add_result(self, name: str, operation: str, time_ms: float, tokens: int = None):
        """Add a benchmark result."""
        if name not in self.results:
            self.results[name] = {}
        if operation not in self.results[name]:
            self.results[name][operation] = []

        result = {"time_ms": time_ms}
        if tokens is not None:
            result["tokens_per_ms"] = tokens / time_ms
        self.results[name][operation].append(result)

    def print_results(self):
        """Print formatted benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        for name, ops in self.results.items():
            print(f"\n{name}:")
            print("-" * 40)
            for op, measurements in ops.items():
                times = [m["time_ms"] for m in measurements]
                avg_time = np.mean(times)
                std_time = np.std(times)

                if "tokens_per_ms" in measurements[0]:
                    tps = [m["tokens_per_ms"] for m in measurements]
                    avg_tps = np.mean(tps)
                    print(f"  {op:20s}: {avg_time:8.2f} ± {std_time:6.2f} ms ({avg_tps:8.0f} tokens/ms)")
                else:
                    print(f"  {op:20s}: {avg_time:8.2f} ± {std_time:6.2f} ms")

    def save_json(self, path: Path):
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {path}")

    def plot_results(self, save_path: Path = None, show: bool = True):
        """Create and display a visualization of benchmark results."""
        # Prepare data for plotting
        tokenizers = []
        encode_times = []
        encode_tps = []  # tokens per ms
        decode_times = []

        for name, ops in self.results.items():
            tokenizers.append(name)

            # Get encode time and throughput
            encode_ops = [op for op in ops.keys() if "encode" in op]
            if encode_ops:
                encode_op = encode_ops[0]
                measurements = ops[encode_op]
                times = [m["time_ms"] for m in measurements]
                encode_times.append(np.mean(times))

                tps = [m.get("tokens_per_ms", 0) for m in measurements]
                encode_tps.append(np.mean(tps) if tps else 0)
            else:
                encode_times.append(0)
                encode_tps.append(0)

            # Get decode time
            decode_ops = [op for op in ops.keys() if "decode" in op]
            if decode_ops:
                decode_op = decode_ops[0]
                measurements = ops[decode_op]
                times = [m["time_ms"] for m in measurements]
                decode_times.append(np.mean(times))
            else:
                decode_times.append(0)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Tokenizer Performance Comparison', fontsize=16, fontweight='bold')

        # Color scheme
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
        color_map = {name: colors[i % len(colors)] for i, name in enumerate(tokenizers)}

        # Plot 1: Encoding Time (lower is better)
        bars1 = axes[0].bar(tokenizers, encode_times, color=[color_map[t] for t in tokenizers], alpha=0.8, edgecolor='black')
        axes[0].set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        axes[0].set_title('Encoding Time\n(lower is better)', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0].set_ylim(bottom=0)

        # Add value labels on bars
        for bar, time_val in zip(bars1, encode_times):
            height = bar.get_height()
            if height > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{time_val:.2f}ms', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Throughput (higher is better)
        bars2 = axes[1].bar(tokenizers, encode_tps, color=[color_map[t] for t in tokenizers], alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Tokens per ms', fontsize=12, fontweight='bold')
        axes[1].set_title('Encoding Throughput\n(higher is better)', fontsize=12)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1].set_ylim(bottom=0)

        # Add value labels on bars
        for bar, tps_val in zip(bars2, encode_tps):
            height = bar.get_height()
            if height > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(tps_val)}', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Decoding Time (lower is better)
        bars3 = axes[2].bar(tokenizers, decode_times, color=[color_map[t] for t in tokenizers], alpha=0.8, edgecolor='black')
        axes[2].set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        axes[2].set_title('Decoding Time\n(lower is better)', fontsize=12)
        axes[2].grid(axis='y', alpha=0.3, linestyle='--')
        axes[2].set_ylim(bottom=0)

        # Add value labels on bars
        for bar, time_val in zip(bars3, decode_times):
            height = bar.get_height()
            if height > 0:
                axes[2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{time_val:.2f}ms', ha='center', va='bottom', fontweight='bold')

        # Rotate x-axis labels for better readability
        for ax in axes:
            ax.tick_params(axis='x', rotation=15, labelsize=11)
            ax.tick_params(axis='y', labelsize=10)

        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")

        # Show the plot if requested (and if DISPLAY is available)
        if show:
            try:
                # Try to detect if we're in a graphical environment
                if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
                    # Re-import with interactive backend for showing
                    import matplotlib
                    matplotlib.use('TkAgg', force=True)
                    import matplotlib.pyplot as plt_new
                    fig_new, axes_new = plt_new.subplots(1, 3, figsize=(16, 5))

                    # Recreate the plots with the new backend
                    for i, (ax, data, title, ylabel, color_fn) in enumerate([
                        (axes_new[0], encode_times, 'Encoding Time\n(lower is better)', 'Time (ms)', lambda x: x),
                        (axes_new[1], encode_tps, 'Encoding Throughput\n(higher is better)', 'Tokens per ms', lambda x: x),
                        (axes_new[2], decode_times, 'Decoding Time\n(lower is better)', 'Time (ms)', lambda x: x),
                    ]):
                        bars = ax.bar(tokenizers, data, color=[color_map[t] for t in tokenizers], alpha=0.8, edgecolor='black')
                        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
                        ax.set_title(title, fontsize=12)
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        ax.set_ylim(bottom=0)

                        for bar, val in zip(bars, data):
                            height = bar.get_height()
                            if height > 0:
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                       f'{val:.2f}' if i != 1 else f'{int(val)}',
                                       ha='center', va='bottom', fontweight='bold')
                        ax.tick_params(axis='x', rotation=15, labelsize=11)
                        ax.tick_params(axis='y', labelsize=10)

                    fig_new.suptitle('Tokenizer Performance Comparison', fontsize=16, fontweight='bold')
                    plt_new.tight_layout()
                    plt_new.show()
                else:
                    print("\nNote: No display detected. Plot saved to file only.")
            except Exception as e:
                print(f"\nNote: Could not display plot interactively: {e}")
                print(f"Plot saved to {save_path}")

        plt.close(fig)


def read_test_data(path: Path) -> str:
    """Read test data from file."""
    with open(path, "r") as f:
        return f.read()


def generate_test_data(words: int) -> str:
    """Generate synthetic test data."""
    # Use a mix of common English words
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
    ]
    return " ".join(np.random.choice(common_words, words))


class BeepeTokenizer:
    """Beepe tokenizer (native Python bindings)."""

    def __init__(self, model_path: Path):
        import beepe
        self.tokenizer = beepe.Tokenizer.load(str(model_path))

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens)


class TiktokenTokenizer:
    """Tiktoken tokenizer wrapper."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        import tiktoken
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.encoding.decode(tokens)


class SentencepieceTokenizer:
    """Sentencepiece tokenizer wrapper."""

    def __init__(self, model_path: str):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.decode(tokens)


def train_sentencepiece_model(text: str, vocab_size: int, model_path: str) -> None:
    """Train a sentencepiece model."""
    import sentencepiece as spm

    # Write text to temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(text)
        temp_path = f.name

    # Train model
    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
    )

    # Clean up
    os.unlink(temp_path)
    os.unlink(f"{model_path}.vocab")


def benchmark_encoding(
    tokenizer, name: str, text: str, iterations: int, results: BenchmarkResults
) -> Tuple[List[int], float]:
    """Benchmark encoding operation."""
    times = []
    tokens = None

    for _ in range(iterations):
        start = time.perf_counter()
        tokens = tokenizer.encode(text)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    results.add_result(name, f"encode ({len(text)} chars)", avg_time, len(tokens))

    return tokens, avg_time


def benchmark_decoding(
    tokenizer, name: str, tokens: List[int], iterations: int, results: BenchmarkResults
):
    """Benchmark decoding operation."""
    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        text = tokenizer.decode(tokens)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    avg_time = np.mean(times)
    results.add_result(name, f"decode ({len(tokens)} tokens)", avg_time)


def main():
    parser = argparse.ArgumentParser(description="Benchmark tokenizers")
    parser.add_argument(
        "--test-data",
        type=Path,
        help="Path to test data file (will generate random data if not provided)",
    )
    parser.add_argument(
        "--beepe-model",
        type=Path,
        help="Path to trained beepe model (will train temporary model if not provided)",
    )
    parser.add_argument(
        "--tiktoken-model",
        default="cl100k_base",
        help="Tiktoken model name (default: cl100k_base)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30000,
        help="Vocabulary size for training (default: 30000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="benchmark_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--skip-beepe",
        action="store_true",
        help="Skip beepe benchmarking",
    )
    parser.add_argument(
        "--skip-tiktoken",
        action="store_true",
        help="Skip tiktoken benchmarking",
    )
    parser.add_argument(
        "--skip-sentencepiece",
        action="store_true",
        help="Skip sentencepiece benchmarking",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't show the plot (useful for headless environments)",
    )

    args = parser.parse_args()

    # Read or generate test data
    if args.test_data and args.test_data.exists():
        print(f"Reading test data from {args.test_data}")
        text = read_test_data(args.test_data)
    else:
        print("Generating synthetic test data (10,000 words)")
        text = generate_test_data(10000)

    print(f"Test data: {len(text)} characters")

    results = BenchmarkResults()

    # Build beepe CLI for training if needed
    if not args.skip_beepe and args.beepe_model is None:
        print("Building beepe CLI in release mode (for training)...")
        subprocess.run(
            ["cargo", "build", "--release", "-p", "beepe-cli"],
            check=True,
        )
        beepe_cli = Path("./target/release/beepe")

    # Train beepe model if not provided
    beepe_model = args.beepe_model
    if not args.skip_beepe:
        if beepe_model is None:
            print("Training temporary beepe model...")
            with tempfile.TemporaryDirectory() as tmpdir:
                beepe_model = Path(tmpdir) / "beepe_model"
                train_data = tempfile.NamedTemporaryFile(mode="w", delete=False)
                train_data.write(text)
                train_data.close()

                subprocess.run(
                    [
                        str(beepe_cli),
                        "train",
                        "--input",
                        train_data.name,
                        "--output",
                        str(beepe_model),
                        "--vocab-size",
                        str(args.vocab_size),
                    ],
                    check=True,
                    capture_output=True,
                )

                os.unlink(train_data.name)

                # Benchmark beepe (inside tempdir so model stays alive)
                print("\nBenchmarking beepe...")
                beepe = BeepeTokenizer(beepe_model)

                # Warmup
                for _ in range(args.warmup):
                    beepe.encode(text)
                    tokens = beepe.encode(text)
                    beepe.decode(tokens)

                # Benchmark
                tokens, _ = benchmark_encoding(
                    beepe, "beepe", text, args.iterations, results
                )
                benchmark_decoding(beepe, "beepe", tokens, args.iterations, results)
        else:
            # Benchmark beepe with provided model
            print("\nBenchmarking beepe...")
            beepe = BeepeTokenizer(beepe_model)

            # Warmup
            for _ in range(args.warmup):
                beepe.encode(text)
                tokens = beepe.encode(text)
                beepe.decode(tokens)

            # Benchmark
            tokens, _ = benchmark_encoding(
                beepe, "beepe", text, args.iterations, results
            )
            benchmark_decoding(beepe, "beepe", tokens, args.iterations, results)

    # Benchmark tiktoken
    if not args.skip_tiktoken:
        print("\nBenchmarking tiktoken...")
        tiktoken = TiktokenTokenizer(args.tiktoken_model)

        # Warmup
        for _ in range(args.warmup):
            tiktoken.encode(text)
            tokens = tiktoken.encode(text)
            tiktoken.decode(tokens)

        # Benchmark
        tokens, _ = benchmark_encoding(
            tiktoken, "tiktoken", text, args.iterations, results
        )
        benchmark_decoding(tiktoken, "tiktoken", tokens, args.iterations, results)

    # Benchmark sentencepiece
    if not args.skip_sentencepiece:
        print("\nBenchmarking sentencepiece...")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "sp_model"
            print("Training sentencepiece model...")
            train_sentencepiece_model(text, args.vocab_size, str(model_path))

            sentencepiece = SentencepieceTokenizer(str(model_path) + ".model")

            # Warmup
            for _ in range(args.warmup):
                sentencepiece.encode(text)
                tokens = sentencepiece.encode(text)
                sentencepiece.decode(tokens)

            # Benchmark
            tokens, _ = benchmark_encoding(
                sentencepiece, "sentencepiece", text, args.iterations, results
            )
            benchmark_decoding(
                sentencepiece, "sentencepiece", tokens, args.iterations, results
            )

    # Print and save results
    results.print_results()
    results.save_json(args.output)

    # Plot and display results
    plot_path = args.output.with_suffix(".png")
    results.plot_results(save_path=plot_path, show=not args.no_plot)


if __name__ == "__main__":
    main()
