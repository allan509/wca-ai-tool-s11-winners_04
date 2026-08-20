"""
llm.py
------

Handles communication between Boma Yetu AI Assistant
and the language model API.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


DEFAULT_MODEL = "gpt-5-mini"


def create_client() -> OpenAI:
    """
    Create an OpenAI API client using the API key
    stored in the environment.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    return OpenAI(api_key=api_key)


def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send a prompt to the language model and return
    the generated response.

    Parameters
    ----------
    prompt:
        Complete prompt containing the system instructions,
        retrieved knowledge, and user question.

    model:
        Model used for generation.

    Returns
    -------
    str
        Generated response.
    """

    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")

    if not prompt.strip():
        raise ValueError("prompt cannot be empty.")

    client = create_client()

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text.strip()