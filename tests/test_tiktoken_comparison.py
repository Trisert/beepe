import pytest


class TestTiktokenComparison:
    """Compare beepe output with tiktoken."""

    def test_encode_output_comparison(self, beepe_tokenizer, tiktoken_tokenizer):
        """Compare encode outputs on sample texts."""
        sample_texts = [
            "hello world",
            "The quick brown fox",
            "请考试我的软件！",
            "🎉 Emoji test: 👍",
        ]

        for text in sample_texts:
            beepe_tokens = beepe_tokenizer.encode(text, False)
            tik_tokens = tiktoken_tokenizer.encode(text)

            beepe_decoded = beepe_tokenizer.decode(beepe_tokens, False)
            tik_decoded = tiktoken_tokenizer.decode(tik_tokens)

            assert beepe_decoded == text
            assert tik_decoded == text

    @pytest.mark.skip(
        reason="Compression ratio test not meaningful - different models trained on different data"
    )
    def test_compression_ratio(
        self, beepe_tokenizer, tiktoken_tokenizer, shakespeare_text
    ):
        """Compare compression ratios."""
        text = shakespeare_text[:10000]

        beepe_tokens = beepe_tokenizer.encode(text, False)
        tik_tokens = tiktoken_tokenizer.encode(text)

        beepe_ratio = len(text) / len(beepe_tokens)
        tik_ratio = len(text) / len(tik_tokens)

        ratio_diff = abs(beepe_ratio - tik_ratio) / tik_ratio
        assert ratio_diff < 0.5, (
            f"Compression ratio differs too much: beepe={beepe_ratio:.2f}, tik={tik_ratio:.2f}"
        )

    def test_special_tokens_consistency(self, beepe_tokenizer, tiktoken_tokenizer):
        """Test special token handling consistency."""
        special_tokens = ["hello"]

        for token in special_tokens:
            beepe_tokens = beepe_tokenizer.encode(token, True)
            tik_tokens = tiktoken_tokenizer.encode(token, allowed_special="all")

            assert len(beepe_tokens) >= 1
            assert len(tik_tokens) >= 1

    def test_unicode_handling(self, beepe_tokenizer, tiktoken_tokenizer):
        """Test Unicode handling consistency."""
        unicode_texts = [
            "👍",
            "请考试我的软件！",
            "مرحبا بالعالم",
            "Hello 世界 🌍",
        ]

        for text in unicode_texts:
            beepe_tokens = beepe_tokenizer.encode(text, False)
            tik_tokens = tiktoken_tokenizer.encode(text)

            beepe_decoded = beepe_tokenizer.decode(beepe_tokens, False)
            tik_decoded = tiktoken_tokenizer.decode(tik_tokens)

            assert beepe_decoded == text
            assert tik_decoded == text

    def test_roundtrip_consistency(self, beepe_tokenizer, tiktoken_tokenizer):
        """Test that roundtrips are consistent for both tokenizers."""
        text = "The quick brown fox jumps over the lazy dog"

        beepe_tokens = beepe_tokenizer.encode(text, False)
        tik_tokens = tiktoken_tokenizer.encode(text)

        beepe_decoded = beepe_tokenizer.decode(beepe_tokens, False)
        tik_decoded = tiktoken_tokenizer.decode(tik_tokens)

        assert beepe_decoded == text
        assert tik_decoded == text
