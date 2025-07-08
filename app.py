import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors.listwise_rerank import LLMListwiseRerank

from langchain.retrievers import ContextualCompressionRetriever
import os
import requests 
from urllib.parse import urlparse, parse_qs
import re
from langchain.prompts import PromptTemplate
import logging
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
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
    
# LLM Initialization
llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash", 
        google_api_key=GOOGLE_API_KEY
    )


# Step 1: Extracting Youtube ID
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


# Step 2: Get Transcript
@st.cache_data(show_spinner="📄 Fetching transcript...")
def get_transcript(video_id):
    session = requests.Session()

    # ScraperAPI Proxy format (rotates IP, adds headers)
    session.proxies = {
        "http": "http://51.79.50.31:9300",
        "https": "http://51.79.50.31:9300"
    }

    # Override youtube_transcript_api internal session
    YouTubeTranscriptApi._session = session

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([d["text"] for d in transcript_list])
    except Exception as e:
        # Can't use st.error inside cached function, so just raise
        raise RuntimeError(f"❌ Could not fetch transcript: {e}")


# Step 3: Embed & store
@st.cache_resource(show_spinner="🔗 Building retriever...")
def create_embeddings(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript)
    docs = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store , len(docs)


# Step 4: RAG chain
def retrieve_documents(vector_store, question, chunk_count):
    #retriever = vector_store.as_retriever(search_type="similarity")
    #chunk_count = len(vector_store.docstore._dict)

    if chunk_count >= 8:
        search_type = "mmr"
        search_kwargs = {"k": 8, "lambda_mult": 0.6}
    else:
        search_type = "similarity"
        search_kwargs = {"k": 4}

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs),
        llm=llm
    )
    #reranker = LLMReranker.from_llm(llm)
    reranker = LLMListwiseRerank.from_llm(llm=llm, top_n=4)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor= reranker,
        base_retriever=retriever,
    )
        
    
    retrieved_docs = compression_retriever.get_relevant_documents(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def build_prompt(context_text, question):
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    return prompt.invoke({"context": context_text, "question": question})

def generate_response(prompt_text):
    ans=  llm.invoke(prompt_text)
    return ans.content

def run_rag_chain(vector_store, question: str, chunk_count: int):
    context_text = retrieve_documents(vector_store, question, chunk_count)
    prompt_text = build_prompt(context_text, question)
    response = generate_response(prompt_text)
    return response, context_text


# Streamlit UI
video_url = st.text_input(
    "📺 Enter YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
)

if video_url:
    try:
        # video_id = video_url.split("v=")[-1].split("&")[0]
        video_id = extract_youtube_id(video_url)
        try:
            transcript = get_transcript(video_id)
        except Exception as e:
            st.error(str(e))
            st.stop()
        st.success("Transcript loaded successfully!")

        vector_store ,chunk_count = create_embeddings(transcript)
        #qa_chain = get_qa_chain(vector_store)
        
       

        question = st.text_input("❓ Ask a question about the video:")
        if question:
            response, context_text = run_rag_chain(vector_store, question, chunk_count)
            with st.spinner("🤖 Generating answer..."):
                st.markdown("### ✅ Answer:")
                st.text(response)

                with st.expander("📚 Relevant Transcript Chunks"):
                    for doc in context_text.split("\n\n"):
                        st.write(doc.strip())

                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        
        




    
    