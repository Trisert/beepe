import pytest


class TestAPICompatibility:
    """Test that beepe API is compatible with tiktoken."""

    def test_encode_signature(self, beepe_tokenizer, tiktoken_tokenizer):
        """Test encode method signature matches."""
        text = "hello world"

        beepe_tokens = beepe_tokenizer.encode(text, False)
        tik_tokens = tiktoken_tokenizer.encode(text)

        assert isinstance(beepe_tokens, list)
        assert isinstance(tik_tokens, list)
        assert all(isinstance(t, int) for t in beepe_tokens)
        assert all(isinstance(t, int) for t in tik_tokens)

    def test_decode_signature(self, beepe_tokenizer, tiktoken_tokenizer):
        """Test decode method signature matches."""
        text = "hello world"
        beepe_tokens = beepe_tokenizer.encode(text, False)
        tik_tokens = tiktoken_tokenizer.encode(text)

        beepe_decoded = beepe_tokenizer.decode(beepe_tokens, False)
        tik_decoded = tiktoken_tokenizer.decode(tik_tokens)

        assert beepe_decoded == text
        assert tik_decoded == text

    def test_vocab_size_property(self, beepe_tokenizer, tiktoken_tokenizer):
        """Test vocab_size property matches."""
        assert hasattr(beepe_tokenizer, "vocab_size")
        assert hasattr(tiktoken_tokenizer, "n_vocab")

        beepe_vocab = beepe_tokenizer.vocab_size()
        tik_vocab = tiktoken_tokenizer.n_vocab

        assert isinstance(beepe_vocab, int)
        assert isinstance(tik_vocab, int)
        assert beepe_vocab > 0
        assert tik_vocab > 0

    def test_batch_encode_signature(self, beepe_tokenizer, tiktoken_tokenizer):
        """Test batch encode signature matches."""
        texts = ["hello", "world", "test"]

        beepe_result = beepe_tokenizer.encode_batch(texts, False)
        tik_result = tiktoken_tokenizer.encode_batch(texts)

        assert len(beepe_result) == len(tik_result)
        assert all(isinstance(r, list) for r in beepe_result)
        assert all(isinstance(r, list) for r in tik_result)

    def test_encode_returns_list_of_ints(self, beepe_tokenizer):
        """Test that encode returns list of integers."""
        text = "hello world"
        tokens = beepe_tokenizer.encode(text, False)

        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_decode_returns_string(self, beepe_tokenizer):
        """Test that decode returns string."""
        tokens = beepe_tokenizer.encode("hello world", False)
        decoded = beepe_tokenizer.decode(tokens, False)

        assert isinstance(decoded, str)
        assert len(decoded) > 0
