# 🎥 VidWise AI — YouTube Video Q&A Chatbot

VidWise AI is an intelligent chatbot built with **Streamlit**, **LangChain**, **Gemini 1.5**, and **FAISS**, allowing you to ask natural language questions about the content of any YouTube video. It extracts the video transcript, semantically indexes it, and uses Gemini 1.5 Flash to answer your questions contextually.

---

## 🚀 Features

- ✅ Extracts transcripts from any YouTube video
- ✅ Splits transcript into meaningful chunks
- ✅ Generates semantic embeddings with MiniLM
- ✅ Fast retrieval using FAISS vector store
- ✅ Uses Gemini 1.5 Flash for intelligent responses
- ✅ Easy-to-use Streamlit UI

---

## 🧠 How It Works

```text
YouTube URL → Transcript → Chunking → Embeddings → FAISS Vector Store
                                                        ↓
                               Question → Similar Chunks → Gemini 1.5 → Answer
