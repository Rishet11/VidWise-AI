import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
import requests

# Config
st.set_page_config(page_title="VidWise AI", layout="centered")
st.title("🎥 VidWise AI: YouTube Q&A Bot")
st.markdown("Ask questions about any YouTube video!")

# Set API Key
GOOGLE_API_KEY = "your_api_key"
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = st.text_input("🔑 Enter your Google API Key", type="password")

# Step 1: Get Transcript
def get_transcript(video_id):
    proxy_url = "http://135.148.120.6:80"
    
    session = requests.Session()
    session.proxies = {
        "http": proxy_url,
        "https": proxy_url
    }

    YouTubeTranscriptApi._session = session

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([d["text"] for d in transcript_list])
    except Exception as e:
        st.error(f"❌ Could not fetch transcript: {e}")
        return None


# Step 2: Embed & store
def build_retriever(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(transcript)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

# Step 3: QA chain
def get_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_retriever(),  # ✅ Proper retriever interface
    return_source_documents=True
    )
    return qa_chain

# --- Streamlit UI ---
video_url = st.text_input("📺 Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo")

if video_url:
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        with st.spinner("📄 Fetching transcript..."):
            transcript = get_transcript(video_id)
        st.success("Transcript loaded successfully!")

        retriever = build_retriever(transcript)
        qa_chain = get_qa_chain(retriever)

        question = st.text_input("❓ Ask a question about the video:")
        if question:
            with st.spinner("🤖 Generating answer..."):
                response = qa_chain(question)
                st.markdown("### ✅ Answer:")
                st.write(response["result"])

                with st.expander("📚 Relevant Transcript Chunks"):
                    for doc in response["source_documents"]:
                        st.write(doc.page_content)
    except Exception as e:
        st.error(f"Error: {str(e)}")
