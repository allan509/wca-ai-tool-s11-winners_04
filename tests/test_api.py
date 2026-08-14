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

def test_chat():
    """
    Test that the chat endpoint accepts a valid question.
    """

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
        == "Received your question: Who can register for Boma Yangu?"
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