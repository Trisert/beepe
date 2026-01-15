# Test fixtures for beepe tokenizer tests

import pytest

SAMPLE_TEXT = """Hello, world! This is a test of the beepe tokenizer.
请考试我的软件！12345
Emoji test: 👍 🚀 🦀
"""

# Basic encoding test cases (expected outputs)
BASIC_ENCODINGS = {
    "hello": [31373],
    "world": [995],
    "hello world": [31373, 995],
}

# Repetitive sequences for testing
REPETITIVE_SEQUENCES = {
    "zeros": ["0", "00", "000", "0000", "00000", "000000"],
    "spaces": [" ", "  ", "   ", "    "],
    "newlines": ["\n", "\n\n", "\n\n\n"],
}

# Unicode test cases
UNICODE_TEST_CASES = [
    "👍",
    "请考试我的软件！12345",
    "مرحبا بالعالم",
    "Hello 世界 🌍",
    "🎉 Emoji test: 👍 🚀 🦀",
    "\ud83d\udc4d",  # Surrogate pair
]

# Special tokens (if configured)
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
]

# Long text for large file tests
LONG_TEXT_PATTERNS = {
    "repeated_zeros": "0" * 1000,
    "mixed_content": "Hello " * 1000 + "World" * 1000,
}

# Test strings for roundtrip tests
ROUNDTRIP_STRINGS = [
    "hello",
    "hello ",
    "hello  ",
    " hello",
    " hello ",
    " hello  ",
    "hello world",
    "请考试我的软件！12345",
    "The quick brown fox jumps over the lazy dog",
    "Hello, world! This is a test.",
]


@pytest.fixture(scope="session")
def sample_text():
    """Return sample text for testing."""
    return SAMPLE_TEXT


@pytest.fixture(scope="session")
def beepe_tokenizer():
    """Create a trained beepe tokenizer."""
    import beepe

    tokenizer = beepe.Tokenizer.builder().vocab_size(30000).min_frequency(2).build()
    tokenizer.train(SAMPLE_TEXT)
    return tokenizer


@pytest.fixture(scope="session")
def tiktoken_tokenizer():
    """Create tiktoken tokenizer for comparison."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        pytest.skip("tiktoken not installed")


@pytest.fixture(scope="session")
def shakespeare_text():
    """Load Shakespeare text for testing."""
    with open("shakespeare.txt", "r") as f:
        return f.read()[:10000]  # First 10k chars


@pytest.fixture
def performance_baseline():
    """Performance baselines from initial benchmarks."""
    return {
        "encode_speed_tokens_per_ms": 25000,
        "decode_speed_tokens_per_ms": 30000,
        "max_slowdown_factor": 2.0,
    }
