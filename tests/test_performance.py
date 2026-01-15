import pytest
import time


@pytest.mark.slow
class TestPerformanceRegression:
    """Performance regression tests."""

    def test_encode_performance(self, beepe_tokenizer, performance_baseline):
        """Test encoding performance meets baseline."""
        text = "hello world " * 1000

        warmup_iterations = 10
        for _ in range(warmup_iterations):
            beepe_tokenizer.encode(text, False)

        measure_iterations = 100
        start = time.time()
        for _ in range(measure_iterations):
            tokens = beepe_tokenizer.encode(text, False)
        elapsed = (time.time() - start) * 1000

        avg_tokens = len(tokens)
        tokens_per_ms = (avg_tokens * measure_iterations) / elapsed

        baseline = performance_baseline["encode_speed_tokens_per_ms"]
        max_slowdown = performance_baseline["max_slowdown_factor"]

        assert tokens_per_ms >= baseline / max_slowdown, (
            f"Encoding too slow: {tokens_per_ms:.0f} tokens/ms < {baseline / max_slowdown:.0f} baseline"
        )

    def test_decode_performance(self, beepe_tokenizer, performance_baseline):
        """Test decoding performance meets baseline."""
        text = "hello world " * 1000
        tokens = beepe_tokenizer.encode(text, False)

        warmup_iterations = 10
        for _ in range(warmup_iterations):
            beepe_tokenizer.decode(tokens, False)

        measure_iterations = 100
        start = time.time()
        for _ in range(measure_iterations):
            decoded = beepe_tokenizer.decode(tokens, False)
        elapsed = (time.time() - start) * 1000

        avg_tokens = len(tokens)
        tokens_per_ms = (avg_tokens * measure_iterations) / elapsed

        baseline = performance_baseline["decode_speed_tokens_per_ms"]
        max_slowdown = performance_baseline["max_slowdown_factor"]

        assert tokens_per_ms >= baseline / max_slowdown, (
            f"Decoding too slow: {tokens_per_ms:.0f} tokens/ms < {baseline / max_slowdown:.0f} baseline"
        )

    def test_batch_encode_performance(self, beepe_tokenizer):
        """Test batch encoding performance."""
        texts = ["hello world"] * 100

        warmup_iterations = 5
        for _ in range(warmup_iterations):
            beepe_tokenizer.encode_batch(texts, False)

        measure_iterations = 50
        start = time.time()
        for _ in range(measure_iterations):
            results = beepe_tokenizer.encode_batch(texts, False)
        elapsed = (time.time() - start) * 1000

        assert elapsed < 1000, f"Batch encoding too slow: {elapsed:.0f}ms"

    def test_large_text_handling(self, beepe_tokenizer):
        """Test handling of large texts."""
        text = "x" * 999_999
        tokens = beepe_tokenizer.encode(text, False)
        assert len(tokens) > 0

        decoded = beepe_tokenizer.decode(tokens, False)
        assert len(decoded) == 999_999

    def test_memory_usage_estimate(self, beepe_tokenizer):
        """Test memory usage is reasonable for operations."""
        import tracemalloc

        tracemalloc.start()

        text = "hello world " * 1000
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak < 100 * 1024 * 1024, (
            f"Memory usage too high: {peak / 1024 / 1024:.1f} MB"
        )
