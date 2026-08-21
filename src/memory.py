"""
memory.py
---------
Conversation memory for the Boma Yetu AI Assistant.

Provides:
1. ConversationMemory
   - Stores messages for one conversation.

2. MemoryManager
   - Manages multiple conversations using conversation IDs.
"""

from datetime import datetime, timezone


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Stores messages exchanged between a user and the assistant.
    """

    def __init__(self, max_messages: int = 20):
        """
        Create a new conversation memory.

        Args:
            max_messages:
                Maximum number of messages to retain.
        """

        if not isinstance(max_messages, int):
            raise TypeError("max_messages must be an integer")

        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than 0"
            )

        self.max_messages = max_messages
        self.messages = []

        self.created_at = datetime.now(
            timezone.utc
        )

        self.updated_at = self.created_at

    # ========================================================
    # ADD MESSAGE
    # ========================================================

    def add_message(
        self,
        role: str,
        content: str,
    ):
        """
        Add a message to the conversation.
        """

        if role not in (
            "user",
            "assistant",
        ):
            raise ValueError(
                "role must be 'user' or 'assistant'"
            )

        if not isinstance(content, str):
            raise TypeError(
                "content must be a string"
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "content cannot be empty"
            )

        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        # Keep the most recent messages.
        if len(self.messages) > self.max_messages:

            self.messages = self.messages[
                -self.max_messages:
            ]

        self.updated_at = datetime.now(
            timezone.utc
        )

    # ========================================================
    # USER MESSAGE
    # ========================================================

    def add_user_message(
        self,
        content: str,
    ):
        """Add a user message."""

        self.add_message(
            "user",
            content,
        )

    # ========================================================
    # ASSISTANT MESSAGE
    # ========================================================

    def add_assistant_message(
        self,
        content: str,
    ):
        """Add an assistant message."""

        self.add_message(
            "assistant",
            content,
        )

    # ========================================================
    # GET MESSAGES
    # ========================================================

    def get_messages(self) -> list[dict]:
        """
        Return a copy of the conversation history.
        """

        return [
            message.copy()
            for message in self.messages
        ]

    # ========================================================
    # GET RECENT MESSAGES
    # ========================================================

    def get_recent_messages(
        self,
        count: int = 5,
    ) -> list[dict]:
        """
        Return the most recent messages.
        """

        if not isinstance(count, int):
            raise TypeError(
                "count must be an integer"
            )

        if count <= 0:
            raise ValueError(
                "count must be greater than 0"
            )

        return [
            message.copy()
            for message
            in self.messages[-count:]
        ]

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:
        """Return the number of stored messages."""

        return len(self.messages)

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(self):
        """Delete all conversation history."""

        self.messages.clear()

        self.updated_at = datetime.now(
            timezone.utc
        )


# ============================================================
# MEMORY MANAGER
# ============================================================

class MemoryManager:
    """
    Manages multiple ConversationMemory objects.

    Each conversation is identified by a unique
    conversation_id.
    """

    def __init__(
        self,
        max_messages: int = 20,
    ):
        if not isinstance(max_messages, int):
            raise TypeError(
                "max_messages must be an integer"
            )

        if max_messages <= 0:
            raise ValueError(
                "max_messages must be greater than 0"
            )

        self.max_messages = max_messages

        self.conversations = {}

    # ========================================================
    # GET OR CREATE
    # ========================================================

    def get_or_create(
        self,
        conversation_id: str,
    ) -> ConversationMemory:
        """
        Return an existing conversation or create
        a new one.
        """

        if not isinstance(
            conversation_id,
            str,
        ):
            raise TypeError(
                "conversation_id must be a string"
            )

        conversation_id = conversation_id.strip()

        if not conversation_id:
            raise ValueError(
                "conversation_id cannot be empty"
            )

        if conversation_id not in self.conversations:

            self.conversations[
                conversation_id
            ] = ConversationMemory(
                max_messages=self.max_messages
            )

        return self.conversations[
            conversation_id
        ]

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        conversation_id: str,
    ):
        """
        Return a conversation if it exists.

        Returns:
            ConversationMemory or None.
        """

        return self.conversations.get(
            conversation_id
        )

    # ========================================================
    # CLEAR
    # ========================================================

    def clear(
        self,
        conversation_id: str,
    ) -> bool:
        """
        Delete a conversation.

        Returns:
            True if deleted, otherwise False.
        """

        if conversation_id in self.conversations:

            del self.conversations[
                conversation_id
            ]

            return True

        return False

    # ========================================================
    # COUNT
    # ========================================================

    def count(self) -> int:
        """Return the number of active conversations."""

        return len(self.conversations)

    # ========================================================
    # CLEAR ALL
    # ========================================================

    def clear_all(self):
        """Delete all conversations."""

        self.conversations.clear()