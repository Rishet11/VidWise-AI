from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory

from config.secrets import GOOGLE_API_KEY

if GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY
    )
else:
    llm = None

memory = ConversationBufferMemory(return_messages=True)
