import streamlit as st
from langchain.schema import HumanMessage, AIMessage

def message_alignment_style():
    """
    Injects custom CSS for chat message alignment and styling,
    with support for both light and dark themes.
    """
    st.markdown("""
        <style>
            /* General chat container */
            .chat-container {
                display: flex;
                flex-direction: column;
            }
            .chat-row {
                display: flex;
                margin-bottom: 1rem;
            }
            .message-bubble {
                padding: 0.8rem 1rem;
                border-radius: 1.2rem;
                max-width: 80%;
                word-wrap: break-word;
            }

            /* User (right-aligned) messages */
            .user-message {
                justify-content: flex-end;
            }
            .user-bubble {
                background-color: #2b313e; /* A neutral dark blue for user */
                color: #ffffff;
                border-bottom-right-radius: 0;
            }

            /* AI (left-aligned) messages */
            .ai-message {
                justify-content: flex-start;
            }
            .ai-bubble {
                /* Uses Streamlit's native background color for secondary elements */
                background-color: var(--secondary-background-color);
                color: var(--text-color);
                border: 1px solid var(--secondary-background-color);
                border-bottom-left-radius: 0;
            }
        </style>
    """, unsafe_allow_html=True)

def show_history(chat_memory, summary_text=""):
    """
    Displays the chat history with right-left alignment for user and AI messages.
    """
    message_alignment_style()
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for msg in chat_memory.buffer:
        if isinstance(msg, AIMessage) and msg.content == summary_text:
            continue
        if isinstance(msg, HumanMessage):
            st.markdown(f"""
                <div class="chat-row user-message">
                    <div class="message-bubble user-bubble">
                        <b>You:</b><br>
                        {msg.content}
                    </div>
                </div>
            """, unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f"""
                <div class="chat-row ai-message">
                    <div class="message-bubble ai-bubble">
                        <b>VidWise AI:</b><br>
                        {msg.content}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def show_context_chunks(context_text):
    with st.expander("📚 Relevant Transcript Chunks"):
        for doc in context_text.split("\n\n"):
            st.write(doc.strip())