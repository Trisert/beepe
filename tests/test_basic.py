import pytest
import sys
from pathlib import Path


class TestBasicEncoding:
    """Test basic encoding and decoding functionality."""

    def test_simple_encode_decode(self, beepe_tokenizer):
        """Test simple encode/decode roundtrip."""
        text = "hello world"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_empty_string(self, beepe_tokenizer):
        """Test encoding empty string."""
        tokens = beepe_tokenizer.encode("", False)
        assert tokens == []

    def test_multiple_whitespace(self, beepe_tokenizer):
        """Test handling of multiple whitespace characters."""
        text = "hello  world   test"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_vocab_size(self, beepe_tokenizer):
        """Test vocab_size method."""
        assert beepe_tokenizer.vocab_size() > 0

    def test_single_word(self, beepe_tokenizer):
        """Test encoding single word."""
        text = "hello"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text
        assert len(tokens) > 0

    def test_punctuation(self, beepe_tokenizer):
        """Test encoding punctuation."""
        text = "Hello, world! How are you?"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_numbers(self, beepe_tokenizer):
        """Test encoding numbers."""
        text = "12345 67890"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text
