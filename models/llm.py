from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from config.secrets import GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

memory = ConversationBufferMemory(return_messages=True)
