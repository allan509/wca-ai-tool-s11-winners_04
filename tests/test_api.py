"""
test_api.py
-----------

Tests for the Boma Yetu AI Assistant API.

These tests check that:
    1. The API application starts correctly.
    2. The home endpoint works.
    3. The health endpoint works.
    4. The chat endpoint accepts a valid question.
    5. The chat endpoint rejects an empty question.
"""


from fastapi.testclient import TestClient

from src.api import app


# ============================================================
# CREATE TEST CLIENT
# ============================================================

client = TestClient(app)


# ============================================================
# TEST HOME ENDPOINT
# ============================================================

def test_home():
    """
    Test that the root endpoint is working.
    """

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Boma Yetu AI Assistant API is running"
    }


# ============================================================
# TEST HEALTH ENDPOINT
# ============================================================

def test_health_check():
    """
    Test that the health-check endpoint returns
    a healthy status.
    """

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy"
    }


# ============================================================
# TEST CHAT ENDPOINT
# ============================================================

def test_chat(monkeypatch):
    """
    Test that the chat endpoint accepts a valid question
    and passes it to the chatbot.
    """

    def fake_answer_with_llm(
        question,
        store,
        top_k=3,
        conversation_history=None,
    ):
        assert question == "Who can register for Boma Yangu?"
        assert store is not None
        assert top_k == 3
        assert conversation_history is not None

        return (
            "Based on the Boma Yangu knowledge base, "
            "eligible Kenyan citizens can register."
        )

    monkeypatch.setattr(
        "src.api.answer_with_llm",
        fake_answer_with_llm,
    )

    response = client.post(
        "/chat",
        json={
            "question": "Who can register for Boma Yangu?"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data

    assert (
        data["answer"]
        == (
            "Based on the Boma Yangu knowledge base, "
            "eligible Kenyan citizens can register."
        )
    )


# ============================================================
# TEST EMPTY QUESTION
# ============================================================

def test_empty_question():
    """
    Test that an empty question is rejected.
    """

    response = client.post(
        "/chat",
        json={
            "question": ""
        }
    )

    assert response.status_code == 400

    assert response.json() == {
        "detail": "Question cannot be empty."
    }


# ============================================================
# TEST MISSING QUESTION
# ============================================================

def test_missing_question():
    """
    Test that a request without the question field
    is rejected by Pydantic validation.
    """

    response = client.post(
        "/chat",
        json={}
    )

    assert response.status_code == 422

def test_chat_conversation_memory(monkeypatch):
    """
    Test that two requests using the same conversation_id
    share conversation memory.
    """

    captured_histories = []

    def fake_answer_with_llm(
        question,
        store,
        top_k=3,
        conversation_history=None,
    ):
        captured_histories.append(
            conversation_history
        )

        if question == "What is Boma Yangu?":
            return "Boma Yangu is an affordable housing programme."

        if question == "How do I register?":
            assert conversation_history is not None

            assert any(
                message["role"] == "user"
                and message["content"] == "What is Boma Yangu?"
                for message in conversation_history
            )

            assert any(
                message["role"] == "assistant"
                and message["content"]
                == "Boma Yangu is an affordable housing programme."
                for message in conversation_history
            )

            return "You can register through the Boma Yangu portal."

        return "Unexpected question."

    monkeypatch.setattr(
        "src.api.answer_with_llm",
        fake_answer_with_llm,
    )

    # --------------------------------------------------------
    # First question
    # --------------------------------------------------------

    first_response = client.post(
        "/chat",
        json={
            "question": "What is Boma Yangu?"
        },
    )

    assert first_response.status_code == 200

    first_data = first_response.json()

    assert "answer" in first_data
    assert "conversation_id" in first_data

    conversation_id = first_data["conversation_id"]

    assert conversation_id

    # --------------------------------------------------------
    # Second question using the same conversation
    # --------------------------------------------------------

    second_response = client.post(
        "/chat",
        json={
            "question": "How do I register?",
            "conversation_id": conversation_id,
        },
    )

    assert second_response.status_code == 200

    second_data = second_response.json()

    assert (
        second_data["answer"]
        == "You can register through the Boma Yangu portal."
    )

    assert (
        second_data["conversation_id"]
        == conversation_id
    )

    # --------------------------------------------------------
    # Confirm both calls were made
    # --------------------------------------------------------

    assert len(captured_histories) == 2

    # First request has no previous conversation.
    assert captured_histories[0] == []

    # Second request has the first exchange.
    assert len(captured_histories[1]) == 2