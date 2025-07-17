import streamlit as st
from langchain.schema import HumanMessage, AIMessage
import streamlit as st

def show_history(chat_memory, summary_text=""):
    
    for msg in chat_memory.buffer:
        # Skip the summary message if it matches the current summary
        if isinstance(msg, AIMessage) and msg.content == summary_text:
            continue
        if isinstance(msg, HumanMessage):
            st.markdown(f"**You:** {msg.content}")
        elif isinstance(msg, AIMessage):
            st.markdown(f"**VidWise AI:** {msg.content}")
            
            
def show_context_chunks(context_text):
    with st.expander("📚 Relevant Transcript Chunks"):
        for doc in context_text.split("\n\n"):
            st.write(doc.strip())