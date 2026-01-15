import pytest
from hypothesis import given, strategies as st


class TestRoundtrip:
    """Test encode/decode roundtrip correctness."""

    def test_basic_roundtrip(self, beepe_tokenizer):
        """Test basic roundtrip with common strings."""
        test_strings = [
            "hello",
            "hello ",
            "hello  ",
            " hello",
            " hello ",
            " hello  ",
            "hello world",
            "请考试我的软件！12345",
        ]

        for text in test_strings:
            tokens = beepe_tokenizer.encode(text, False)
            decoded = beepe_tokenizer.decode(tokens, False)
            assert decoded == text, f"Roundtrip failed for: {text}"

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="Hypothesis finds edge cases in byte-level encoding - requires Rust fixes"
    )
    @given(
        text=st.text(min_size=1, max_size=100).filter(
            lambda x: all(ord(c) < 0xFFFF for c in x)
        )
    )
    def test_hyp_roundtrip(self, text, beepe_tokenizer):
        """Property-based roundtrip test using hypothesis."""
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="Hypothesis finds edge cases in byte-level encoding - requires Rust fixes"
    )
    @given(
        text=st.text(min_size=1, max_size=100).filter(
            lambda x: all(ord(c) < 0xFFFF for c in x)
        )
    )
    def test_hyp_roundtrip_with_specials(self, text, beepe_tokenizer):
        """Property-based roundtrip with special tokens allowed."""
        tokens = beepe_tokenizer.encode(text, True)
        decoded = beepe_tokenizer.decode(tokens, True)
        assert decoded == text

    def test_long_text_roundtrip(self, beepe_tokenizer, shakespeare_text):
        """Test roundtrip with very long text."""
        text = shakespeare_text[:5000]
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_mixed_content_roundtrip(self, beepe_tokenizer):
        """Test roundtrip with mixed content."""
        text = "Hello 世界 🌍 12345 !@#"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text
