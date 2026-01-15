import pytest


class TestUnicode:
    """Test Unicode and byte encoding."""

    @pytest.mark.parametrize(
        "text",
        [
            "👍",
            "请考试我的软件！12345",
            "مرحبا بالعالم",
            "Hello 世界 🌍",
            "🎉 Emoji test: 👍 🚀 🦀",
        ],
    )
    def test_unicode_roundtrip(self, text, beepe_tokenizer):
        """Test Unicode roundtrip."""
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_emoji_encoding(self, beepe_tokenizer):
        """Test emoji encoding."""
        text = "👍"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text
        assert len(tokens) >= 1

    def test_chinese_characters(self, beepe_tokenizer):
        """Test Chinese character encoding."""
        text = "请考试我的软件！"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text
        assert len(tokens) > 0

    def test_arabic_text(self, beepe_tokenizer):
        """Test Arabic text encoding."""
        text = "مرحبا بالعالم"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_mixed_unicode(self, beepe_tokenizer):
        """Test mixed Unicode content."""
        text = "Hello 世界 🌍 مرحبا"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_special_symbols(self, beepe_tokenizer):
        """Test special symbols."""
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        tokens = beepe_tokenizer.encode(text, False)
        decoded = beepe_tokenizer.decode(tokens, False)
        assert decoded == text

    def test_unicode_normalization(self, beepe_tokenizer):
        """Test Unicode normalization handling."""
        text1 = "café"
        text2 = "cafe\u0301"  # e + combining acute accent
        tokens1 = beepe_tokenizer.encode(text1, False)
        tokens2 = beepe_tokenizer.encode(text2, False)
        # Both should encode to valid tokens
        assert len(tokens1) > 0
        assert len(tokens2) > 0
