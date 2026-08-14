import pytest

from src import llm


def test_default_model():
    """Test that a default model is configured."""

    assert isinstance(llm.DEFAULT_MODEL, str)
    assert llm.DEFAULT_MODEL.strip() != ""


def test_create_client_without_api_key(monkeypatch):
    """Test that missing API key raises an error."""

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    with pytest.raises(ValueError):
        llm.create_client()


def test_generate_response_invalid_prompt_type():
    """Test that prompt must be a string."""

    with pytest.raises(TypeError):
        llm.generate_response(123)


def test_generate_response_empty_prompt():
    """Test that an empty prompt is rejected."""

    with pytest.raises(ValueError):
        llm.generate_response("")


def test_generate_response_whitespace_prompt():
    """Test that whitespace-only prompts are rejected."""

    with pytest.raises(ValueError):
        llm.generate_response("   ")

def test_generate_response_with_mock(monkeypatch):
    """Test LLM response generation without calling the real API."""

    class FakeResponse:
        output_text = "Boma Yangu is a platform for affordable housing."

    class FakeResponses:
        def create(self, model, input):
            return FakeResponse()

    class FakeClient:
        responses = FakeResponses()

    monkeypatch.setattr(
        llm,
        "create_client",
        lambda: FakeClient(),
    )

    response = llm.generate_response(
        "What is Boma Yangu?"
    )

    assert response == (
        "Boma Yangu is a platform for affordable housing."
    )