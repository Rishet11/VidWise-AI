from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors.listwise_rerank import LLMListwiseRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage
from models.llm import llm
import streamlit as st

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

def run_rag_chain(vector_store, question: str, chunk_count: int, memory):
    context_text = retrieve_documents(vector_store, question, chunk_count)
    prompt_text = build_prompt(context_text, question, memory.buffer)
    response = generate_response(prompt_text)

    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response)

    return response, context_text