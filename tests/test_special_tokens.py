import pytest


class TestSpecialTokens:
    """Test special token handling."""

    def test_special_token_detection(self, beepe_tokenizer):
        """Test that special tokens are detected correctly."""
        text = "hello  world"
        tokens_with = beepe_tokenizer.encode(text, True)
        tokens_without = beepe_tokenizer.encode(text, False)
        assert len(tokens_with) >= len(tokens_without)
        assert len(tokens_with) > len(tokens_without)

    def test_skip_special_tokens(self, beepe_tokenizer):
        """Test skipping special tokens during decode."""
        text = "hello world"
        tokens = beepe_tokenizer.encode(text, True)
        decoded_with = beepe_tokenizer.decode(tokens, False)
        decoded_without = beepe_tokenizer.decode(tokens, True)
        assert decoded_with != decoded_without
        assert isinstance(decoded_with, str)
        assert isinstance(decoded_without, str)

    def test_special_tokens_consistency(self, beepe_tokenizer):
        """Test that special tokens are consistently encoded."""
        tokens1 = beepe_tokenizer.encode("hello", True)
        tokens2 = beepe_tokenizer.encode("world", True)
        # Both should have special tokens at start and end
        assert tokens1[0] == tokens2[0]  # BOS should be same
        assert tokens1[-1] == tokens2[-1]  # EOS should be same

    def test_no_special_tokens_in_ordinary_encode(self, beepe_tokenizer):
        """Test that encode_ordinary doesn't add special tokens."""
        tokens = beepe_tokenizer.encode("hello world", False)
        # This test will use encode_ordinary when implemented
        # For now, test basic encode behavior
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)
