"""
Automated tests for the memory module.
"""

import pytest

from src.memory import ConversationMemory

# Creation

def test_create_memory():
    """Check that a new conversation memory starts empty."""

    memory = ConversationMemory()

    assert memory.count() == 0
    assert memory.get_messages() == []


def test_create_memory_with_custom_limit():
    """Check that a custom maximum message limit is accepted."""

    memory = ConversationMemory(max_messages=10)

    assert memory.max_messages == 10
    assert memory.count() == 0

# Initialization validation
def test_invalid_max_messages_type():
    """Check that max_messages must be an integer."""

    with pytest.raises(TypeError):
        ConversationMemory(max_messages="20")


def test_invalid_max_messages_value():
    """Check that max_messages must be greater than zero."""

    with pytest.raises(ValueError):
        ConversationMemory(max_messages=0)


def test_negative_max_messages():
    """Check that negative limits are rejected."""

    with pytest.raises(ValueError):
        ConversationMemory(max_messages=-5)

# Add Message

def test_add_user_message():
    """Check that a user message can be stored."""

    memory = ConversationMemory()

    memory.add_message(
        "user",
        "What is Boma Yangu?",
    )

    assert memory.count() == 1

    messages = memory.get_messages()

    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is Boma Yangu?"


def test_add_assistant_message():
    """Check that an assistant message can be stored."""

    memory = ConversationMemory()

    memory.add_message(
        "assistant",
        "Boma Yangu is a housing portal.",
    )

    assert memory.count() == 1

    messages = memory.get_messages()

    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == (
        "Boma Yangu is a housing portal."
    )

# Convenience methods

def test_add_user_message_method():
    """Check the add_user_message convenience method."""

    memory = ConversationMemory()

    memory.add_user_message(
        "Who qualifies for affordable housing?"
    )

    assert memory.count() == 1
    assert memory.get_messages()[0]["role"] == "user"


def test_add_assistant_message_method():
    """Check the add_assistant_message convenience method."""

    memory = ConversationMemory()

    memory.add_assistant_message(
        "Please check the eligibility information."
    )

    assert memory.count() == 1
    assert memory.get_messages()[0]["role"] == "assistant"

# Message validation

def test_invalid_role():
    """Check that only user and assistant roles are allowed."""

    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.add_message(
            "system",
            "Some message",
        )


def test_invalid_content_type():
    """Check that message content must be a string."""

    memory = ConversationMemory()

    with pytest.raises(TypeError):
        memory.add_message(
            "user",
            123,
        )


def test_empty_content():
    """Check that empty messages are rejected."""

    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.add_message(
            "user",
            "",
        )


def test_whitespace_content():
    """Check that whitespace-only messages are rejected."""

    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.add_message(
            "user",
            "   ",
        )


def test_message_content_is_stripped():
    """Check that unnecessary whitespace is removed."""

    memory = ConversationMemory()

    memory.add_user_message(
        "   What is Boma Yangu?   "
    )

    assert memory.get_messages()[0]["content"] == (
        "What is Boma Yangu?"
    )

# Multiple messages

def test_multiple_messages():
    """Check that several messages are stored in order."""

    memory = ConversationMemory()

    memory.add_user_message("Hello")

    memory.add_assistant_message(
        "Hello! How can I help you?"
    )

    memory.add_user_message(
        "What is Boma Yangu?"
    )

    assert memory.count() == 3

    messages = memory.get_messages()

    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"

# Get messages

def test_get_messages_returns_copy():
    """
    Check that modifying the returned list does not directly
    modify the internal memory list.
    """

    memory = ConversationMemory()

    memory.add_user_message("Hello")

    messages = memory.get_messages()

    messages.clear()

    assert memory.count() == 1

# Recent messages

def test_get_recent_messages():
    """Check that recent messages can be retrieved."""

    memory = ConversationMemory()

    memory.add_user_message("Message 1")
    memory.add_assistant_message("Message 2")
    memory.add_user_message("Message 3")

    recent = memory.get_recent_messages(2)

    assert len(recent) == 2
    assert recent[0]["content"] == "Message 2"
    assert recent[1]["content"] == "Message 3"


def test_get_recent_messages_more_than_available():
    """
    Requesting more messages than currently stored should
    return all available messages.
    """

    memory = ConversationMemory()

    memory.add_user_message("Message 1")
    memory.add_user_message("Message 2")

    recent = memory.get_recent_messages(10)

    assert len(recent) == 2


def test_invalid_recent_count_type():
    """Check that recent count must be an integer."""

    memory = ConversationMemory()

    with pytest.raises(TypeError):
        memory.get_recent_messages("5")


def test_invalid_recent_count_value():
    """Check that recent count must be greater than zero."""

    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.get_recent_messages(0)

# Maximum message limit

def test_max_messages_limit():
    """
    Check that old messages are removed when the maximum
    message limit is reached.
    """

    memory = ConversationMemory(max_messages=3)

    memory.add_user_message("Message 1")
    memory.add_user_message("Message 2")
    memory.add_user_message("Message 3")
    memory.add_user_message("Message 4")

    assert memory.count() == 3

    messages = memory.get_messages()

    assert messages[0]["content"] == "Message 2"
    assert messages[1]["content"] == "Message 3"
    assert messages[2]["content"] == "Message 4"


def test_message_limit_preserves_latest_messages():
    """Check that only the newest messages are retained."""

    memory = ConversationMemory(max_messages=2)

    memory.add_user_message("Old message")
    memory.add_user_message("Recent message 1")
    memory.add_user_message("Recent message 2")

    messages = memory.get_messages()

    assert len(messages) == 2
    assert messages[0]["content"] == "Recent message 1"
    assert messages[1]["content"] == "Recent message 2"

# Clear memory

def test_clear_memory():
    """Check that clear removes all conversation history."""

    memory = ConversationMemory()

    memory.add_user_message("Hello")

    memory.add_assistant_message(
        "Hello! How can I help?"
    )

    assert memory.count() == 2

    memory.clear()

    assert memory.count() == 0
    assert memory.get_messages() == []


def test_clear_empty_memory():
    """Check that clearing an already empty memory is safe."""

    memory = ConversationMemory()

    memory.clear()

    assert memory.count() == 0
def test_conversation_history_is_used_in_prompt():
    from src.chatbot import build_prompt

    history = [
        {
            "role": "user",
            "content": (
                "What housing projects are available in Kiambu?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Projects include developments in Kikuyu and Ruiru."
            ),
        },
    ]

    prompt = build_prompt(
        question="Which one is in Kikuyu?",
        documents=[],
        conversation_history=history,
    )

    assert "CONVERSATION HISTORY:" in prompt
    assert "What housing projects are available in Kiambu?" in prompt
    assert (
        "Projects include developments in Kikuyu and Ruiru."
        in prompt
    )
    assert "Which one is in Kikuyu?" in prompt    