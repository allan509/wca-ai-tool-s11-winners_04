"""
memory.py
---------
Conversation memory for the Boma Yetu AI Assistant.

This first version stores conversation history in memory (RAM).

Later, this module can be connected to SQLite for persistent
conversation storage.
"""


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:
    """
    Stores messages exchanged between a user and the assistant.

    Each message has:
        - role: "user" or "assistant"
        - content: the message text
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
            raise ValueError("max_messages must be greater than 0")

        self.max_messages = max_messages
        self.messages = []

    # ========================================================
    # ADD MESSAGE
    # ========================================================

    def add_message(self, role: str, content: str):
        """
        Add a message to the conversation.

        Args:
            role:
                Either "user" or "assistant".

            content:
                Message text.
        """

        if role not in ("user", "assistant"):
            raise ValueError(
                "role must be 'user' or 'assistant'"
            )

        if not isinstance(content, str):
            raise TypeError("content must be a string")

        content = content.strip()

        if not content:
            raise ValueError("content cannot be empty")

        message = {
            "role": role,
            "content": content,
        }

        self.messages.append(message)

        # Keep only the most recent messages.
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[
                -self.max_messages:
            ]

    # ========================================================
    # ADD USER MESSAGE
    # ========================================================

    def add_user_message(self, content: str):
        """
        Add a message from the user.
        """

        self.add_message("user", content)

    # ========================================================
    # ADD ASSISTANT MESSAGE
    # ========================================================

    def add_assistant_message(self, content: str):
        """
        Add a message from the assistant.
        """

        self.add_message("assistant", content)

    # ========================================================
    # GET MESSAGES
    # ========================================================

    def get_messages(self) -> list[dict]:
        """
        Return all stored conversation messages.

        Returns:
            A copy of the message list.
        """

        return self.messages.copy()

    # ========================================================
    # GET RECENT MESSAGES
    # ========================================================

    def get_recent_messages(
        self,
        count: int = 5,
    ) -> list[dict]:
        """
        Return the most recent messages.

        Args:
            count:
                Number of recent messages to return.

        Returns:
            List containing recent messages.
        """

        if not isinstance(count, int):
            raise TypeError("count must be an integer")

        if count <= 0:
            raise ValueError("count must be greater than 0")

        return self.messages[-count:]

    # ========================================================
    # COUNT MESSAGES
    # ========================================================

    def count(self) -> int:
        """
        Return the number of stored messages.
        """

        return len(self.messages)

    # ========================================================
    # CLEAR MEMORY
    # ========================================================

    def clear(self):
        """
        Delete all conversation history.
        """

        self.messages.clear()