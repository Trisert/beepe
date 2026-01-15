import pytest


class TestErrorHandling:
    """Test error handling."""

    def test_text_too_large(self, beepe_tokenizer):
        """Test that very large text raises error."""
        text = "x" * 1_000_001
        with pytest.raises(Exception):
            beepe_tokenizer.encode(text, False)

    def test_empty_token_list(self, beepe_tokenizer):
        """Test decoding empty token list."""
        decoded = beepe_tokenizer.decode([], False)
        assert decoded == ""

    def test_invalid_token_ids(self, beepe_tokenizer):
        """Test decoding with invalid token IDs."""
        # Very large token ID should return an error string
        decoded = beepe_tokenizer.decode([999999999], False)
        assert "<invalid token>" in decoded or "999999999" in decoded

    def test_decode_with_special_tokens_skipped(self, beepe_tokenizer):
        """Test decoding with special tokens skipped."""
        tokens = beepe_tokenizer.encode("hello world", True)
        # Should not raise when decoding with skip_special_tokens=True
        decoded = beepe_tokenizer.decode(tokens, True)
        assert isinstance(decoded, str)

    def test_encode_with_special_tokens_false(self, beepe_tokenizer):
        """Test encoding with special tokens disabled."""
        text = "hello world"
        tokens = beepe_tokenizer.encode(text, False)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_encode_with_special_tokens_true(self, beepe_tokenizer):
        """Test encoding with special tokens enabled."""
        text = "hello world"
        tokens = beepe_tokenizer.encode(text, True)
        assert isinstance(tokens, list)
        assert len(tokens) >= 2  # At least BOS and EOS
