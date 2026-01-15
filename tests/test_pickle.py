import pytest
import pickle


@pytest.mark.skip(
    reason="Pickle serialization requires proper class registration - see PyO3 pickle limitations"
)
class TestPickle:
    """Test pickle serialization."""

    def test_pickle_roundtrip(self, beepe_tokenizer):
        """Test pickle serialization/deserialization."""
        pickled = pickle.dumps(beepe_tokenizer)
        unpickled = pickle.loads(pickled)

        text = "hello world"
        tokens = unpickled.encode(text, False)
        decoded = unpickled.decode(tokens, False)
        assert decoded == text

    def test_pickle_preserves_vocab_size(self, beepe_tokenizer):
        """Test that pickle preserves vocabulary size."""
        original_size = beepe_tokenizer.vocab_size()

        pickled = pickle.dumps(beepe_tokenizer)
        unpickled = pickle.loads(pickled)

        assert unpickled.vocab_size() == original_size

    def test_pickle_with_different_texts(self, beepe_tokenizer):
        """Test pickle with various texts."""
        texts = ["hello", "world", "test 123", "请考试我的软件！"]

        pickled = pickle.dumps(beepe_tokenizer)
        unpickled = pickle.loads(pickled)

        for text in texts:
            tokens = unpickled.encode(text, False)
            decoded = unpickled.decode(tokens, False)
            assert decoded == text

    def test_pickle_preserves_batch_functionality(self, beepe_tokenizer):
        """Test that pickle preserves batch functionality."""
        texts = ["hello", "world", "test"]

        pickled = pickle.dumps(beepe_tokenizer)
        unpickled = pickle.loads(pickled)

        batch_result = unpickled.encode_batch(texts, False)
        assert len(batch_result) == len(texts)
