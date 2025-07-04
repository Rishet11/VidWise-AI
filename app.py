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
from urllib.parse import urlparse, parse_qs
import re
from dotenv import load_dotenv
load_dotenv()

# Config
st.set_page_config(page_title="VidWise AI", layout="centered")
st.title("🎥 VidWise AI: YouTube Q&A Bot")
st.markdown("Ask questions about any YouTube video!")



# First, try to load from Streamlit Cloud secrets
GOOGLE_API_KEY = None
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    # If not available (e.g., local), fallback to .env
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
try:
    SCRAPERAPI_KEY = st.secrets.get("SCRAPERAPI_KEY")
except Exception:
    SCRAPERAPI_KEY = os.getenv("SCRAPERAPI_KEY")
    
    



#Step 1: Extracting Youtube ID
def extract_youtube_id(url):
    parsed = urlparse(url)

    # Case 1: youtu.be short URL
    if parsed.netloc == "youtu.be":
        return parsed.path.lstrip("/")

    # Case 2: youtube.com/watch?v=...
    if "youtube.com" in parsed.netloc:
        query_params = parse_qs(parsed.query)
        if "v" in query_params:
            return query_params["v"][0]

        # Case 3: /embed/VIDEO_ID
        match_embed = re.search(r"/embed/([a-zA-Z0-9_-]{11})", parsed.path)
        if match_embed:
            return match_embed.group(1)

        # Case 4: /shorts/VIDEO_ID
        match_shorts = re.search(r"/shorts/([a-zA-Z0-9_-]{11})", parsed.path)
        if match_shorts:
            return match_shorts.group(1)

    return None  # No valid video ID found

#Step 2: Get Transcript
def get_transcript(video_id):
    session = requests.Session()

    # ScraperAPI Proxy format (rotates IP, adds headers)
    session.proxies = {
        "http": f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001",
        "https": f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"
    }

    # Override youtube_transcript_api internal session
    YouTubeTranscriptApi._session = session

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([d["text"] for d in transcript_list])
    except Exception as e:
        st.error(f"❌ Could not fetch transcript: {e}")
        return None

#Step 3: Embed & store
def build_retriever(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(transcript)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    return db

#Step 4: QA chain
def get_qa_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_retriever(search_kwargs={"k": 3}),  # ✅ Proper retriever interface
    return_source_documents=True
    )
    return qa_chain



#Streamlit UI
video_url = st.text_input("📺 Enter YouTube Video URL:", placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo")

if video_url:
    try:
        #video_id = video_url.split("v=")[-1].split("&")[0]
        video_id = extract_youtube_id(video_url)
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
