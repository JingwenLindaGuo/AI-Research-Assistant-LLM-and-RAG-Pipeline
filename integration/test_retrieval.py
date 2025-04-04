# test_retrieval.py
# test this with pytest installed and using "pytest --maxfail=1 --disable-warnings -q"
import pytest
from integration.retrieval import retrieve_chunks, prepare_prompt, answer_query, conversation_history, document_memory


# Create a fake index object with a query method for testing
class FakeIndex:
    def query(self, vector, top_k, include_metadata, namespace):
        # Simulate a response with one match
        return {
            "matches": [
                {
                    "metadata": {
                        "text": "Test document content about machine learning.",
                        "source": "Test Source"
                    }
                }
            ]
        }

@pytest.fixture
def fake_index():
    return FakeIndex()

def test_retrieve_chunks(fake_index):
    # Clear memory before test
    conversation_history.clear()
    document_memory.clear()
    
    query = "machine learning"
    chunks = retrieve_chunks(query, fake_index, top_k=1)
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert "Test document content" in chunks[0]["text"]

def test_prepare_prompt():
    # Create a scenario with some conversation and document memory
    conversation_history.clear()
    document_memory.clear()
    conversation_history.extend([
        "Query: What is ML?\nAnswer: ML stands for machine learning."
    ])
    document_memory.extend([
        {"text": "Test document content about ML.", "source": "Test Source"}
    ])
    query = "Explain ML in detail."
    prompt = prepare_prompt(query, document_memory)
    assert "Conversation history:" in prompt
    assert "Relevant documents:" in prompt
    assert "Query: Explain ML in detail." in prompt

def test_answer_query(fake_index, monkeypatch):
    # For testing answer_query, we need to simulate the ChatOpenAI call.
    # We'll monkeypatch llm.invoke to return a fixed answer.

    def fake_invoke(self, prompt, *args, **kwargs):
        return "Simulated answer based on prompt: " + prompt

    # Monkey patch ChatOpenAI.invoke method
    monkeypatch.setattr("integration.retrieval.ChatOpenAI.invoke", fake_invoke)
    
    conversation_history.clear()
    document_memory.clear()
    query = "What is ML?"
    answer = answer_query(query, fake_index, top_k=1)
    assert "Simulated answer" in answer
