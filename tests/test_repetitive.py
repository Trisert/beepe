import pytest


class TestRepetitiveContent:
    """Test handling of repetitive content."""

    def test_repeated_zeros(self, beepe_tokenizer):
        """Test encoding of repeated zeros."""
        zeros_list = ["0", "00", "000", "0000", "00000", "000000"]

        for zeros in zeros_list:
            tokens = beepe_tokenizer.encode(zeros, False)
            decoded = beepe_tokenizer.decode(tokens, False)
            assert decoded == zeros

    def test_repeated_spaces(self, beepe_tokenizer):
        """Test encoding of repeated spaces."""
        spaces_list = [" ", "  ", "   ", "    ", "     "]

        for spaces in spaces_list:
            tokens = beepe_tokenizer.encode(spaces, False)
            decoded = beepe_tokenizer.decode(tokens, False)
            assert decoded == spaces

    def test_repeated_newlines(self, beepe_tokenizer):
        """Test encoding of repeated newlines."""
        newlines_list = ["\n", "\n\n", "\n\n\n", "\n\n\n\n"]

        for newlines in newlines_list:
            tokens = beepe_tokenizer.encode(newlines, False)
            decoded = beepe_tokenizer.decode(tokens, False)
            assert decoded == newlines

    @pytest.mark.slow
    def test_catastrophically_repetitive(self, beepe_tokenizer):
        """Test catastrophically repetitive patterns (1k+ chars)."""
        patterns = ["^", "0", "a", " ", "\n"]

        for c in patterns:
            big_value = c * 1000
            assert big_value == beepe_tokenizer.decode(
                beepe_tokenizer.encode(big_value, False), False
            )

            big_value = " " + big_value
            assert big_value == beepe_tokenizer.decode(
                beepe_tokenizer.encode(big_value, False), False
            )

            big_value = big_value + "\n"
            assert big_value == beepe_tokenizer.decode(
                beepe_tokenizer.encode(big_value, False), False
            )

    def test_repeated_words(self, beepe_tokenizer):
        """Test encoding of repeated words."""
        text = "hello world hello world hello world"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_repeated_phrases(self, beepe_tokenizer):
        """Test encoding of repeated phrases."""
        phrase = "the quick brown fox"
        text = " ".join([phrase] * 10)
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text
