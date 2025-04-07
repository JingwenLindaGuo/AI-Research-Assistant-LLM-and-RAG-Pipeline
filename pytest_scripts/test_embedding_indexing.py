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
    monkeypatch.setattr("lgp.group3.backend.Embedder.client", fake_client)
    yield
