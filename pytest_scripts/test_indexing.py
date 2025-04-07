import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import random
from types import SimpleNamespace
import backend.Database  

# Fake Pinecone client that simulates the behavior of Pinecone.
class FakePinecone:
    def __init__(self):
        self.indexes = []          # List to hold index names.
        self.created_indexes = []  # Log of create_index calls.
    
    def list_indexes(self):
        return [{"name": name} for name in self.indexes]
    
    def create_index(self, name, dimension, metric, spec):
        self.indexes.append(name)
        self.created_indexes.append((name, dimension, metric, spec))
    
    def delete_index(self, index_name):
        if index_name in self.indexes:
            self.indexes.remove(index_name)
        else:
            # Simulate behavior when deletion fails.
            raise Exception("Index does not exist")
    def getIndex(self, index_name):
        # Return a fake index instance that captures upsert calls.
        return FakeIndex(index_name)
    
class FakeIndex:
    def __init__(self, name):
        self.name = name
        self.upsert_called = False
        self.upsert_vectors = None
        self.upsert_namespace = None
    
    def upsert(self, vectors, namespace):
        self.upsert_called = True
        self.upsert_vectors = vectors
        self.upsert_namespace = namespace

# fake embed function that simulates embedding by returning the length of the text.
def fake_embed(input):
    data = [SimpleNamespace(embedding=[len(text)]) for text in input]
    return SimpleNamespace(data=data)

@pytest.fixture
def fake_pc(monkeypatch):
    fake = FakePinecone()
    monkeypatch.setattr(backend.Database, "pc", fake)
    return fake

def test_create_index(fake_pc, monkeypatch):
    # Override random.choice to always return "x" so the generated index name is predictable.
    monkeypatch.setattr(random, "choice", lambda seq: "x")
    
    index_name = backend.Database.create_index()
    # Expected index name is 10 "x" characters.
    assert index_name == "xxxxxxxxxx"
    # Verify the index is added to the fake client's indexes.
    assert "xxxxxxxxxx" in fake_pc.indexes
    # Also, check that create_index logged the call.
    assert any(item[0] == "xxxxxxxxxx" for item in fake_pc.created_indexes)