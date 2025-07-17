from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st


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