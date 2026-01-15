import pytest


class TestBatchOperations:
    """Test batch encoding and decoding."""

    def test_batch_encode_single(self, beepe_tokenizer):
        """Test batch encode with single text."""
        text = "hello world"
        batch_result = beepe_tokenizer.encode_batch([text], False)
        single_result = beepe_tokenizer.encode(text, False)
        assert batch_result == [single_result]

    def test_batch_encode_multiple(self, beepe_tokenizer):
        """Test batch encode with multiple texts."""
        texts = ["hello world", "goodbye world", "test"]
        batch_result = beepe_tokenizer.encode_batch(texts, False)

        for text, tokens in zip(texts, batch_result):
            single_tokens = beepe_tokenizer.encode(text, False)
            assert tokens == single_tokens

    def test_empty_batch(self, beepe_tokenizer):
        """Test batch operations with empty list."""
        result = beepe_tokenizer.encode_batch([], False)
        assert result == []

    def test_batch_encode_with_special_tokens(self, beepe_tokenizer):
        """Test batch encode with special tokens."""
        texts = ["hello", "world", "test"]
        batch_result = beepe_tokenizer.encode_batch(texts, True)

        assert len(batch_result) == len(texts)
        for tokens in batch_result:
            assert len(tokens) > 0

    def test_batch_encode_unicode(self, beepe_tokenizer):
        """Test batch encode with unicode text."""
        texts = ["Hello 世界", "مرحبا بالعالم", "👍🚀🦀"]
        batch_result = beepe_tokenizer.encode_batch(texts, False)

        assert len(batch_result) == len(texts)
        for text, tokens in zip(texts, batch_result):
            assert len(tokens) > 0
            decoded = beepe_tokenizer.decode(tokens, False)
            assert decoded == text

    def test_large_batch(self, beepe_tokenizer):
        """Test batch with large number of texts."""
        texts = ["hello world"] * 100
        batch_result = beepe_tokenizer.encode_batch(texts, False)

        assert len(batch_result) == 100
        for tokens in batch_result:
            assert len(tokens) > 0
