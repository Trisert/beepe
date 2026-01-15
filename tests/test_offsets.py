import pytest


class TestOffsets:
    """Test decode_with_offsets functionality."""

    def test_basic_offsets(self, beepe_tokenizer):
        """Test basic offset tracking."""
        text = "hello world"
        tokens = beepe_tokenizer.encode(text, False)

        if hasattr(beepe_tokenizer, "decode_with_offsets"):
            decoded_text, offsets = beepe_tokenizer.decode_with_offsets(tokens)
            assert decoded_text == text
            assert len(offsets) == len(tokens)
            assert offsets[0] == 0
        else:
            pytest.skip("decode_with_offsets not implemented yet")

    def test_multibyte_offsets(self, beepe_tokenizer):
        """Test offsets with multibyte characters."""
        text = "Hello 世界"
        tokens = beepe_tokenizer.encode(text, False)

        if hasattr(beepe_tokenizer, "decode_with_offsets"):
            decoded_text, offsets = beepe_tokenizer.decode_with_offsets(tokens)
            assert decoded_text == text
            assert all(0 <= offset <= len(text) for offset in offsets)
        else:
            pytest.skip("decode_with_offsets not implemented yet")

    def test_unicode_offsets(self, beepe_tokenizer):
        """Test offsets with unicode characters."""
        text = "🎉 Hello 🌍 World"
        tokens = beepe_tokenizer.encode(text, False)

        if hasattr(beepe_tokenizer, "decode_with_offsets"):
            decoded_text, offsets = beepe_tokenizer.decode_with_offsets(tokens)
            assert decoded_text == text
            assert len(offsets) == len(tokens)
        else:
            pytest.skip("decode_with_offsets not implemented yet")
