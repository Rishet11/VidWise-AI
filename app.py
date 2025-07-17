import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors.listwise_rerank import LLMListwiseRerank
from langchain.retrievers import ContextualCompressionRetriever
from deep_translator import GoogleTranslator 
import os
import requests 
from urllib.parse import urlparse, parse_qs
import re
from langchain.prompts import PromptTemplate
import logging
from langchain.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)
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
        google_api_key=GOOGLE_API_KEY,
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

    return None 



# Step 2: Get Transcript
@st.cache_data(show_spinner="📄 Fetching transcript...")
def get_transcript(video_id):
    proxy_url = f"http://scraperapi:{SCRAPERAPI_KEY}@proxy-server.scraperapi.com:8001"

    session = requests.Session()
    session.proxies = {
        "http": proxy_url,
        "https": proxy_url
    }
    
    session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
    })

    # Override youtube_transcript_api internal session
    YouTubeTranscriptApi._session = session
    ip_check = session.get("http://httpbin.org/ip")
    print("Detected IP via ScraperAPI:", ip_check.text)

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([d["text"] for d in transcript_list])
    
    except:
        pass
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["hi"])
        return " ".join([d["text"] for d in transcript_list])
    
    
    except Exception as e:
        raise RuntimeError(f"❌ Could not fetch transcript: {e}")


# Step 3: Embed & store
@st.cache_resource(show_spinner="🔗 Building retriever...")
def create_embeddings(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript)
    docs = [Document(page_content=chunk) for chunk in chunks]

    #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store , len(docs)


def generate_summary(transcript):
    """Generates a summary of the transcript."""
    prompt = f"""
    You are an expert at summarizing YouTube videos. Please provide a concise summary of the following transcript. 
    Focus on the main points and key takeaways. Use bullet points for clarity.

    Transcript:
    {transcript}
    """
    response = llm.invoke(prompt)
    return response.content



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

    #Reranking
    reranker = LLMListwiseRerank.from_llm(llm=llm, top_n=4)
    
    #Contextual Compression 
    compression_retriever = ContextualCompressionRetriever(
        base_compressor= reranker,
        base_retriever=retriever,
    )
        
    
    retrieved_docs = compression_retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def build_prompt(context_text, question, chat_history):
    messages = []
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            messages.append(f"VidWise AI: {msg.content}")
    history_text = "\n".join(messages)
    
    prompt = PromptTemplate(
        template="""
        You are **VidWise AI**, a smart assistant that helps answer questions based on YouTube video transcripts.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say 'This video doesn’t seem to mention that, so I’m not sure.'
        🎯 INSTRUCTIONS:
        - Answer the question directly and clearly.
        - Do NOT say "based on the transcript" or refer to the context explicitly.
        
        🧠 Previous conversation:
        {history_text}
        
        📚 TRANSCRIPT CONTEXT:
        {context}
        Question: {question}
        """,
        input_variables=['history_text', 'context', 'question']
    )

    return prompt.invoke({"history_text": history_text, "context": context_text, "question": question})

def generate_response(prompt_text):
    ans=  llm.invoke(prompt_text)
    return ans.content

def run_rag_chain(vector_store, question: str, chunk_count: int):
    chat_history = st.session_state.chat_memory.buffer
    context_text = retrieve_documents(vector_store, question, chunk_count)
    prompt_text = build_prompt(context_text, question, chat_history)
    response = generate_response(prompt_text)
    
    # Save current turn to memory
    st.session_state.chat_memory.chat_memory.add_user_message(question)
    st.session_state.chat_memory.chat_memory.add_ai_message(response)
    
    return response, context_text


# Streamlit UI
video_url = st.text_input(
    "📺 Enter YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo",
)

if video_url:
    try:

        video_id = extract_youtube_id(video_url)
        
        transcript_placeholder = st.empty()
        try:
            transcript = get_transcript(video_id)
        except Exception as e:
            st.error(str(e))
            st.stop()
       
        #Success message for loading transcript
        with transcript_placeholder.container():
            st.success("✅ Transcript loaded successfully!")

        #Creating Vector Store
        vector_store ,chunk_count = create_embeddings(transcript)
        
        #Removing the Success message
        transcript_placeholder.empty()
        
        
        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = ConversationBufferMemory(return_messages=True)
            
            
        if "summary_generated" not in st.session_state:
            st.session_state.summary_generated = False
            
        if "summary_text" not in st.session_state:
             st.session_state.summary_text = ""
        
        #Show Get Summary button only if summary hasn't been generated
        if not st.session_state.summary_generated:
            if st.button("✨ Get Summary"):
                with st.spinner("✍️ Generating summary..."):
                    summary = generate_summary(transcript)
                    st.session_state.summary_text = summary
                    st.session_state.summary_generated = True
                    st.session_state.chat_memory.chat_memory.add_ai_message(summary)
                    st.rerun()

        # Show summary and regenerate button after summary is generated
        if st.session_state.summary_generated:
            
            
            if st.button("🔄 Regenerate Summary"):
                with st.spinner("♻️ Regenerating..."):
                    new_summary = generate_summary(transcript)
                    st.session_state.summary_text = new_summary
                    
                    # Update memory (replace last message if it's a summary)
                    if (
                        st.session_state.chat_memory.chat_memory.messages and 
                        st.session_state.chat_memory.chat_memory.messages[-1].type == "ai"
                    ):
                        st.session_state.chat_memory.chat_memory.messages.pop()
                    st.session_state.chat_memory.chat_memory.add_ai_message(new_summary)
                    st.rerun()
                    
            #Summary text below regenerate button 
            st.markdown("### 📜 Video Summary:")  
            st.markdown(st.session_state.summary_text)
            
        
        #st.subheader("🧠 Chat with the video")

        # Show past messages
        for msg in st.session_state.chat_memory.buffer:
            # Skip the summary message if it matches the current summary
            if isinstance(msg, AIMessage) and msg.content == st.session_state.summary_text:
                continue
            if isinstance(msg, HumanMessage):
                st.markdown(f"**You:** {msg.content}")
            elif isinstance(msg, AIMessage):
                st.markdown(f"**VidWise AI:** {msg.content}")
        
       

        question = st.text_input("❓ Ask a question about the video:", key= "chat_input")
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