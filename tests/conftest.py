import pytest
import sys
from pathlib import Path

SAMPLE_TEXT = """Hello, world! This is a test of the beepe tokenizer.
请考试我的软件！12345
Emoji test: 👍 🚀 🦀
"""


@pytest.fixture(scope="session")
def sample_text():
    """Return sample text for testing."""
    return SAMPLE_TEXT


@pytest.fixture(scope="session")
def beepe_tokenizer():
    """Create a trained beepe tokenizer."""
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
        return f.read()[:10000]


@pytest.fixture
def performance_baseline():
    """Performance baselines from initial benchmarks."""
    return {
        "encode_speed_tokens_per_ms": 25000,
        "decode_speed_tokens_per_ms": 30000,
        "max_slowdown_factor": 2.0,
    }
