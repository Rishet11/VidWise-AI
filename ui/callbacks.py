import streamlit as st
from core.youtube_utils import extract_youtube_id, get_transcript
from core.embeddings import create_embeddings
from core.summarizer import generate_summary
from core.rag_pipeline import run_rag_chain
from models.llm import memory
from ui.display import show_history, show_context_chunks

def handle_all_events():
    video_url = st.text_input(
        "📺 Enter YouTube Video URL:",
        placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo"
    )
    
    if not video_url:
        return
    
    try:
        video_id = extract_youtube_id(video_url)
        transcript_placeholder = st.empty()
        transcript = get_transcript(video_id)
        
        with transcript_placeholder.container():
            st.success("✅ Transcript loaded successfully!")
        
        vector_store, chunk_count = create_embeddings(transcript)
        transcript_placeholder.empty()
        
        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = memory
            
        if "summary_generated" not in st.session_state:
            st.session_state.summary_generated = False
            
        if "summary_text" not in st.session_state:
            st.session_state.summary_text = ""
            
        if "question_answered_after_summary" not in st.session_state:
            st.session_state.question_answered_after_summary = False
        
        if not st.session_state.summary_generated:
            if st.button("✨ Get Summary"):
                with st.spinner("✍️ Generating summary..."):
                    summary = generate_summary(transcript)
                    st.session_state.summary_text = summary
                    st.session_state.summary_generated = True
                    st.session_state.chat_memory.chat_memory.add_ai_message(summary)
                    st.rerun()
        
        if st.session_state.summary_generated:
            if st.button("🔄 Regenerate Summary"):
                with st.spinner("♻️ Regenerating..."):
                    new_summary = generate_summary(transcript)
                    st.session_state.summary_text = new_summary
                    # Replace last message if it's the summary
                    if (
                        st.session_state.chat_memory.chat_memory.messages and
                        st.session_state.chat_memory.chat_memory.messages[-1].type == "ai"
                    ):
                        st.session_state.chat_memory.chat_memory.messages.pop()
                    st.session_state.chat_memory.chat_memory.add_ai_message(new_summary)
                    st.rerun()
            
            # Expander that closes only after summary is generated AND a question is answered
            expander_expanded = not (st.session_state.summary_generated and st.session_state.question_answered_after_summary)
            with st.expander("### 📜 Video Summary", expanded=expander_expanded):
                st.write(st.session_state.summary_text)
            
            show_history(st.session_state.chat_memory, st.session_state.summary_text)
        
        # Chat input is always shown (moved outside the summary_generated block)
        question = st.chat_input("Ask anything about the video:", key="chat_input")
                
        if question:
            st.session_state.chat_memory.chat_memory.add_user_message(question)
            
            conversation_placeholder = st.empty()
            
            # Show the conversation with just the user question
            with conversation_placeholder.container():
                show_history(st.session_state.chat_memory, st.session_state.summary_text)
            
            with st.spinner("🤖 Generating answer..."):
                response, context_text = run_rag_chain(vector_store, question, chunk_count, st.session_state.chat_memory)
                st.session_state.chat_memory.chat_memory.add_ai_message(response)
                
                with conversation_placeholder.container():
                    show_history(st.session_state.chat_memory, st.session_state.summary_text)
                
                # Set flag to collapse expander only if summary was already generated
                if st.session_state.summary_generated:
                    st.session_state.question_answered_after_summary = True
                
                
    except Exception as e:
        st.error(f"Error: {str(e)}")