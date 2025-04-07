import pytest
from lgp.group3.backend.Embedder import embed, client
from types import SimpleNamespace

class FakeEmbeddings:
    def create(self, input, model):
        # Simulate a response by returning each text's length as the embedding.
        data = [SimpleNamespace(embedding=[len(text)]) for text in input]
        return SimpleNamespace(data=data)

@pytest.fixture(autouse=True)
def dummy_client(monkeypatch):
    # Create and replace the client with a fake one
    fake_client = SimpleNamespace(embeddings=FakeEmbeddings())
    # Client path needs to be changed to the path of the Embedder file on the users pc
    monkeypatch.setattr("lgp.group3.backend.Embedder.client", fake_client)
    yield

def test_embed_with_list():

    test_data = [{"text": "hello"}, {"text": "pytest"}]
    result = embed(test_data)

    # Expect two embeddings: one for "hello" (length 5) and one for "pytest" (length 6)
    assert len(result) == 2
    assert result[0]["values"] == [5]
    assert result[1]["values"] == [6]

def test_embed_with_single_string():

    test_text = "testing"
    result = embed(test_text)
    
    # Expect one embedding: "testing" has a length of 7.
    assert len(result) == 1
    assert result[0]["values"] == [7]