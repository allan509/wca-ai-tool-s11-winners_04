"""

Streamlit user interface for the Boma Yetu AI Assistant.

The frontend communicates with the FastAPI backend rather than
calling the RAG/LLM components directly.

Architecture:

    Streamlit UI
        ↓
    FastAPI /chat
        ↓
    Chatbot / RAG
        ↓
    Knowledge Base
        ↓
    OpenAI
"""

import requests
import streamlit as st


# Configuration

API_URL = "http://127.0.0.1:8000"


# Page Configuration

st.set_page_config(
    page_title="Boma Yetu AI Assistant",
    page_icon="🏠",
    layout="centered",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f5f7fa;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 Boma Yetu AI Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Your AI assistant for information about Kenya's
    Affordable Housing Programme.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About Boma Yetu")

    st.write(
        """
        Boma Yetu AI Assistant helps users find information
        about Kenya's Affordable Housing Programme using a
        trusted document knowledge base.
        """
    )

    st.divider()

    st.subheader("You can ask about")

    st.markdown(
        """
        - 🏠 Housing eligibility
        - 📝 Registration
        - 💰 Housing levy
        - 💳 Payments and savings
        - 📍 Housing projects
        - 🎯 Allocation
        - 📞 Contact information
        """
    )

    st.divider()

    st.caption(
        "Powered by RAG + OpenAI + FastAPI"
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I am the Boma Yetu AI Assistant. "
                "I can help you find information about Kenya's "
                "Affordable Housing Programme.\n\n"
                "What would you like to know?"
            ),
        }
    ]


# ============================================================
# DISPLAY CONVERSATION
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about affordable housing..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

    else:

        # ----------------------------------------------------
        # Display user's question immediately
        # ----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # ----------------------------------------------------
        # Call FastAPI backend
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the Boma Yetu knowledge base..."
            ):

                try:

                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "question": question
                        },
                        timeout=120,
                    )

                    # ------------------------------------------------
                    # Successful response
                    # ------------------------------------------------

                    if response.status_code == 200:

                        data = response.json()

                        answer = data.get(
                            "answer",
                            "",
                        )

                        if not answer:

                            answer = (
                                "The assistant returned an empty "
                                "response."
                            )

                        st.markdown(answer)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )

                    # ------------------------------------------------
                    # API validation error
                    # ------------------------------------------------

                    elif response.status_code == 400:

                        try:

                            detail = response.json().get(
                                "detail",
                                "Invalid question.",
                            )

                        except Exception:

                            detail = "Invalid question."

                        st.error(detail)

                    # ------------------------------------------------
                    # Other API errors
                    # ------------------------------------------------

                    else:

                        st.error(
                            "The Boma Yetu API returned an "
                            f"error (HTTP {response.status_code})."
                        )

                # ----------------------------------------------------
                # Backend unavailable
                # ----------------------------------------------------

                except requests.exceptions.ConnectionError:

                    st.error(
                        "I could not connect to the Boma Yetu "
                        "API. Please make sure the FastAPI server "
                        "is running on http://127.0.0.1:8000."
                    )

                # ----------------------------------------------------
                # Request timeout
                # ----------------------------------------------------

                except requests.exceptions.Timeout:

                    st.error(
                        "The request took too long to complete. "
                        "Please try again."
                    )

                # ----------------------------------------------------
                # Unexpected error
                # ----------------------------------------------------

                except Exception as error:

                    st.error(
                        f"An unexpected error occurred: {error}"
                    )